#!/usr/bin/env python3
"""
Update all top-level Git repositories in the current directory and summarize
code-relevant changes with an OpenAI-compatible chat model.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


OPENAI_API_BASE_ENV = "OPENAI_API_BASE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_OPENAI_BASE = "https://api.deepseek.com"
DEFAULT_OPENAI_MODEL = "deepseek-v4-pro"
MAX_LOG_CHARS = 24000
MAX_FILES_CHARS = 12000
MAX_STAT_CHARS = 8000
SHELL_ENV_TIMEOUT = 8

PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "NO_PROXY",
)
GIT_NETWORK_ENV_KEYS = (
    *PROXY_ENV_KEYS,
    "GIT_PROXY_COMMAND",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "SSH_AUTH_SOCK",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
LLM_ENV_KEYS = (OPENAI_API_BASE_ENV, OPENAI_API_KEY_ENV, OPENAI_MODEL_ENV)
SHELL_IMPORT_ENV_KEYS = (*GIT_NETWORK_ENV_KEYS, *LLM_ENV_KEYS, "PATH")
LANGUAGE_OVERRIDE_ENV = "UPDATE_REPOS_LANG"

_GIT_ENV_CACHE: dict[str, str] | None = None


def is_chinese_locale(value: str) -> bool:
    normalized = value.strip().lower().replace("_", "-")
    return normalized.startswith("zh") or normalized.startswith("chinese")


def detect_macos_language() -> str | None:
    if sys.platform != "darwin":
        return None

    try:
        completed = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    for raw_line in completed.stdout.splitlines():
        language = raw_line.strip().strip('"').strip(",;").strip()
        if not language or language in {"(", ")"}:
            continue
        return "zh" if is_chinese_locale(language) else "en"
    return None


def detect_output_language() -> str:
    override = os.environ.get(LANGUAGE_OVERRIDE_ENV, "")
    if override:
        return "zh" if is_chinese_locale(override) else "en"

    for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(key, "")
        if is_chinese_locale(value):
            return "zh"

    macos_language = detect_macos_language()
    if macos_language:
        return macos_language

    return "en"


OUTPUT_LANGUAGE = detect_output_language()


def ui(zh: str, en: str) -> str:
    return zh if OUTPUT_LANGUAGE == "zh" else en


class GitError(RuntimeError):
    pass


@dataclass
class RepoUpdate:
    name: str
    before: str = ""
    after: str = ""
    status: str = "unknown"
    summary: str = ""
    message: str = ""

    @property
    def has_update(self) -> bool:
        return self.status == "updated"

    @property
    def no_update(self) -> bool:
        return self.status == "no_update"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=ui(
            "更新当前目录下的 Git 仓库，并用 OpenAI 兼容模型总结核心代码变更。",
            "Update Git repositories in the current directory and summarize code changes with an OpenAI-compatible model.",
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help=ui(
            "要遍历的根目录，默认为当前目录。",
            "Root directory to scan. Defaults to the current directory.",
        ),
    )
    parser.add_argument(
        "--ignore-file",
        type=Path,
        default=None,
        help=ui(
            "忽略目录列表文件，默认为 root/git_ignore.txt。",
            "Ignore-list file. Defaults to root/git_ignore.txt.",
        ),
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=300,
        help=ui(
            "单条 git 命令超时时间，单位秒。",
            "Timeout for each git command, in seconds.",
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=90.0,
        help=ui(
            "LLM API 请求超时时间，单位秒。",
            "Timeout for each LLM API request, in seconds.",
        ),
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help=ui(
            "只更新仓库并输出原始提交标题，不调用 LLM。",
            "Update repositories and print raw commit subjects without calling the LLM.",
        ),
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help=ui(
            "手动指定 git/LLM 使用的代理，例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:7890。",
            "Proxy for git and LLM requests, for example http://127.0.0.1:7890 or socks5://127.0.0.1:7890.",
        ),
    )
    parser.add_argument(
        "--no-shell-env",
        action="store_true",
        help=ui(
            "不从登录/交互 shell 补齐代理、SSH 和 PATH 环境变量。",
            "Do not import proxy, SSH, and PATH variables from login/interactive shells.",
        ),
    )
    parser.add_argument(
        "--no-system-proxy",
        action="store_true",
        help=ui(
            "不读取 macOS 系统代理设置。",
            "Do not read macOS system proxy settings.",
        ),
    )
    return parser.parse_args()


def load_ignore_patterns(ignore_file: Path) -> set[str]:
    if not ignore_file.exists():
        return set()

    patterns: set[str] = set()
    for raw_line in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.add(line.rstrip("/"))
    return patterns


def is_ignored(path: Path, patterns: Iterable[str]) -> bool:
    name = path.name
    rel = path.as_posix().rstrip("/")
    for pattern in patterns:
        normalized = pattern.strip().rstrip("/")
        if not normalized:
            continue
        if name == normalized or rel == normalized:
            return True
        if fnmatch.fnmatch(name, normalized) or fnmatch.fnmatch(rel, normalized):
            return True
    return False


def discover_repositories(root: Path, ignore_patterns: set[str]) -> list[Path]:
    repos: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or is_ignored(child, ignore_patterns):
            continue
        if (child / ".git").exists():
            repos.append(child)
    return repos


def short_hash(value: str) -> str:
    return value[:8] if value else ""


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n...(truncated {omitted} characters)"


def parse_env_output(output: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            env[key] = value
    return env


def read_shell_env(shell: str, flags: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            [shell, flags, "env"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=SHELL_ENV_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    return parse_env_output(completed.stdout)


def collect_shell_env() -> dict[str, str]:
    shells: list[str] = []
    for candidate in (os.environ.get("SHELL"), "/bin/zsh", "/bin/bash"):
        if candidate and candidate not in shells and Path(candidate).exists():
            shells.append(candidate)

    collected: dict[str, str] = {}
    for shell in shells:
        for flags in ("-lc", "-lic"):
            shell_env = read_shell_env(shell, flags)
            for key in SHELL_IMPORT_ENV_KEYS:
                value = shell_env.get(key)
                if value and key not in collected:
                    collected[key] = value
    return collected


def read_macos_system_proxy_env() -> dict[str, str]:
    if sys.platform != "darwin":
        return {}

    try:
        completed = subprocess.run(
            ["scutil", "--proxy"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=SHELL_ENV_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}

    settings: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if " : " not in line:
            continue
        key, value = line.split(" : ", 1)
        settings[key.strip()] = value.strip()

    env: dict[str, str] = {}
    if settings.get("HTTPEnable") == "1" and settings.get("HTTPProxy"):
        proxy = f"http://{settings['HTTPProxy']}:{settings.get('HTTPPort', '80')}"
        env["http_proxy"] = proxy
        env["HTTP_PROXY"] = proxy
    if settings.get("HTTPSEnable") == "1" and settings.get("HTTPSProxy"):
        proxy = f"http://{settings['HTTPSProxy']}:{settings.get('HTTPSPort', '443')}"
        env["https_proxy"] = proxy
        env["HTTPS_PROXY"] = proxy
    if settings.get("SOCKSEnable") == "1" and settings.get("SOCKSProxy"):
        proxy = f"socks5://{settings['SOCKSProxy']}:{settings.get('SOCKSPort', '1080')}"
        env["all_proxy"] = proxy
        env["ALL_PROXY"] = proxy
    return env


def normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if "://" in proxy:
        return proxy
    return f"http://{proxy}"


def apply_proxy(env: dict[str, str], proxy: str) -> None:
    proxy = normalize_proxy(proxy)
    for key in ("http_proxy", "https_proxy", "all_proxy"):
        env[key] = proxy
        env[key.upper()] = proxy


def merge_missing_env(env: dict[str, str], extra: dict[str, str]) -> None:
    for key, value in extra.items():
        if not value:
            continue
        if key == "PATH":
            env[key] = value
        elif not env.get(key):
            env[key] = value


def configure_git_env(
    *,
    proxy: str | None = None,
    use_shell_env: bool = True,
    use_system_proxy: bool = True,
) -> None:
    global _GIT_ENV_CACHE

    env = os.environ.copy()
    if use_shell_env:
        merge_missing_env(env, collect_shell_env())
    if use_system_proxy:
        merge_missing_env(env, read_macos_system_proxy_env())
    if proxy:
        apply_proxy(env, proxy)

    env["GIT_TERMINAL_PROMPT"] = "0"
    _GIT_ENV_CACHE = env

    for key in (*GIT_NETWORK_ENV_KEYS, *LLM_ENV_KEYS):
        value = env.get(key)
        if value:
            os.environ[key] = value


def get_git_env() -> dict[str, str]:
    if _GIT_ENV_CACHE is None:
        configure_git_env()
    assert _GIT_ENV_CACHE is not None
    return _GIT_ENV_CACHE.copy()


def run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            env=get_git_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(
            ui(
                f"git {' '.join(args)} 超时",
                f"git {' '.join(args)} timed out",
            )
        ) from exc

    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if not detail:
            detail = ui(
                f"git {' '.join(args)} 失败，退出码 {completed.returncode}",
                f"git {' '.join(args)} failed with exit code {completed.returncode}",
            )
        raise GitError(detail)
    return completed


def git_stdout(repo: Path, args: list[str], *, timeout: int, check: bool = True) -> str:
    return run_git(repo, args, timeout=timeout, check=check).stdout.strip()


def has_ancestor(repo: Path, ancestor: str, descendant: str, *, timeout: int) -> bool:
    result = run_git(
        repo,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        timeout=timeout,
        check=False,
    )
    return result.returncode == 0


def resolve_upstream(repo: Path, *, timeout: int) -> str:
    upstream = git_stdout(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        timeout=timeout,
        check=False,
    )
    if upstream:
        return upstream

    branch = git_stdout(repo, ["branch", "--show-current"], timeout=timeout)
    if not branch:
        raise GitError(
            ui(
                "当前处于 detached HEAD，无法自动判断要跟踪的远端分支",
                "detached HEAD; cannot determine the upstream branch automatically",
            )
        )

    fetch_remote(repo, "origin", timeout=timeout)
    candidate = f"origin/{branch}"
    verify = run_git(
        repo,
        ["rev-parse", "--verify", "--quiet", candidate],
        timeout=timeout,
        check=False,
    )
    if verify.returncode == 0:
        return candidate

    raise GitError(
        ui(
            "未设置 upstream，且找不到对应的 origin 分支",
            "no upstream is configured and no matching origin branch was found",
        )
    )


def fetch_remote(repo: Path, remote: str | None, *, timeout: int) -> None:
    args = ["fetch", "--prune"]
    if remote:
        args.append(remote)
    run_git(repo, args, timeout=timeout)


def fetch_for_upstream(repo: Path, upstream: str, *, timeout: int) -> None:
    remote = None
    if "/" in upstream and not upstream.startswith("@"):
        remote = upstream.split("/", 1)[0]
    fetch_remote(repo, remote, timeout=timeout)


def update_repository(repo: Path, *, git_timeout: int) -> RepoUpdate:
    result = RepoUpdate(name=repo.name)
    try:
        result.before = git_stdout(repo, ["rev-parse", "HEAD"], timeout=git_timeout)
        upstream = resolve_upstream(repo, timeout=git_timeout)
        fetch_for_upstream(repo, upstream, timeout=git_timeout)
        remote_head = git_stdout(repo, ["rev-parse", upstream], timeout=git_timeout)

        if result.before == remote_head:
            result.status = "no_update"
            result.after = result.before
            return result

        if has_ancestor(repo, result.before, remote_head, timeout=git_timeout):
            run_git(
                repo,
                ["merge", "--ff-only", "--autostash", upstream],
                timeout=git_timeout,
            )
            result.after = git_stdout(repo, ["rev-parse", "HEAD"], timeout=git_timeout)
            result.status = "updated" if result.after != result.before else "no_update"
            return result

        if has_ancestor(repo, remote_head, result.before, timeout=git_timeout):
            result.status = "no_update"
            result.after = result.before
            result.message = ui(
                "本地分支领先远端，无远端更新",
                "local branch is ahead of the remote; no remote updates",
            )
            return result

        result.status = "skipped"
        result.after = result.before
        result.message = ui(
            "本地分支与远端分支已分叉，已跳过自动更新",
            "local and remote branches have diverged; automatic update was skipped",
        )
        return result
    except GitError as exc:
        result.status = "error"
        result.after = result.before
        result.message = str(exc).splitlines()[0]
        return result


def collect_change_context(repo: Path, before: str, after: str, *, git_timeout: int) -> dict[str, str]:
    log = git_stdout(
        repo,
        [
            "log",
            "--no-merges",
            "--date=short",
            "--pretty=format:commit %h%nDate: %ad%nSubject: %s%nBody:%n%b%n",
            f"{before}..{after}",
        ],
        timeout=git_timeout,
    )
    if not log:
        log = git_stdout(
            repo,
            [
                "log",
                "--date=short",
                "--pretty=format:commit %h%nDate: %ad%nSubject: %s%nBody:%n%b%n",
                f"{before}..{after}",
            ],
            timeout=git_timeout,
        )

    files = git_stdout(
        repo,
        ["diff", "--name-status", "--find-renames", before, after],
        timeout=git_timeout,
    )
    stat = git_stdout(
        repo,
        ["diff", "--stat", "--find-renames", before, after],
        timeout=git_timeout,
    )
    shortstat = git_stdout(
        repo,
        ["diff", "--shortstat", before, after],
        timeout=git_timeout,
    )

    return {
        "log": truncate(log, MAX_LOG_CHARS),
        "files": truncate(files, MAX_FILES_CHARS),
        "stat": truncate(stat, MAX_STAT_CHARS),
        "shortstat": shortstat,
    }


def fallback_summary(repo: Path, before: str, after: str, *, git_timeout: int) -> str:
    subjects = git_stdout(
        repo,
        ["log", "--no-merges", "--pretty=format:%s", f"{before}..{after}"],
        timeout=git_timeout,
    )
    if not subjects:
        return ui(
            "- 主要更新：无明显核心代码变更。",
            "- Main update: no notable core code changes.",
        )

    lines = [line.strip() for line in subjects.splitlines() if line.strip()]
    lines = lines[:5]
    return "\n".join(f"- {line}" for line in lines)


class OpenAISummarizer:
    def __init__(self, *, api_timeout: float) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                ui(
                    "缺少依赖：openai。请先安装 openai 包。",
                    "Missing dependency: openai. Please install the openai package first.",
                )
            ) from exc

        api_key = os.environ.get(OPENAI_API_KEY_ENV)
        api_base = os.environ.get(OPENAI_API_BASE_ENV, DEFAULT_OPENAI_BASE)
        self.model = os.environ.get(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL)

        if not api_key:
            raise RuntimeError(
                ui(
                    f"缺少 API Key，请设置 {OPENAI_API_KEY_ENV}。",
                    f"Missing API key. Please set {OPENAI_API_KEY_ENV}.",
                )
            )

        client_kwargs = {"api_key": api_key, "timeout": api_timeout}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)

    def summarize(self, repo_name: str, before: str, after: str, context: dict[str, str]) -> str:
        user_prompt = f"""
Repository: {repo_name}
Update range: {short_hash(before)} -> {short_hash(after)}

Commit log:
{context["log"]}

Changed files:
{context["files"]}

Diff stat:
{context["stat"]}

Overall stat:
{context["shortstat"]}
""".strip()

        language_instruction = (
            "The user's terminal locale is Chinese. Write the final summary bullets in Simplified Chinese."
            if OUTPUT_LANGUAGE == "zh"
            else "Write the final summary bullets in English."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer summarizing Git updates. "
                        "Ignore pure CI/CD changes, automated dependency bumps, formatting-only changes, "
                        "documentation-only changes, README edits, version bumps, lockfile churn, "
                        "and configuration cleanup unless they directly affect build, deployment, "
                        "runtime behavior, or user-visible behavior. Focus on implementation code, APIs, "
                        "architecture, performance, features, compatibility, and bug fixes. "
                        "Return 2 to 5 concise Markdown bullets. If there are no meaningful implementation "
                        "changes, return exactly one bullet saying that no notable core code changes were found. "
                        f"{language_instruction}"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return (
            content
            or ui(
                "- 主要更新：无明显核心代码变更。",
                "- Main update: no notable core code changes.",
            )
        ).strip()


def summarize_update(
    repo: Path,
    item: RepoUpdate,
    *,
    summarizer: OpenAISummarizer | None,
    git_timeout: int,
    no_ai: bool,
) -> None:
    if not item.has_update:
        return

    try:
        if no_ai:
            item.summary = fallback_summary(
                repo,
                item.before,
                item.after,
                git_timeout=git_timeout,
            )
            return

        context = collect_change_context(
            repo,
            item.before,
            item.after,
            git_timeout=git_timeout,
        )
        assert summarizer is not None
        item.summary = summarizer.summarize(
            item.name,
            item.before,
            item.after,
            context,
        )
    except Exception as exc:  # Keep the final report useful even if one summary fails.
        detail = str(exc).splitlines()[0]
        item.summary = ui(
            f"- 更新摘要生成失败：{detail}",
            f"- Summary generation failed: {detail}",
        )


def print_report_header(root: Path) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(ui("# Git 仓库更新摘要", "# Git Repository Update Summary"))
    print(ui(f"根目录：{root}", f"Root: {root}"))
    print(ui(f"生成时间：{generated}", f"Generated: {generated}"))
    print()
    sys.stdout.flush()


def print_repo_result(item: RepoUpdate) -> None:
    print(f"## {item.name}")
    if item.has_update:
        print(f"`{short_hash(item.before)} -> {short_hash(item.after)}`")
        print(
            item.summary.strip()
            or ui(
                "- 主要更新：无明显核心代码变更。",
                "- Main update: no notable core code changes.",
            )
        )
    elif item.no_update:
        if item.message:
            print(ui(f"无更新（{item.message}）。", f"No updates ({item.message})."))
        else:
            print(ui("无更新。", "No updates."))
    elif item.status == "skipped":
        print(ui(f"未更新：{item.message}。", f"Not updated: {item.message}."))
    else:
        print(ui(f"更新失败：{item.message}。", f"Update failed: {item.message}."))
    print()
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    ignore_file = args.ignore_file or (root / "git_ignore.txt")

    if not root.exists() or not root.is_dir():
        print(
            ui(
                f"错误：目录不存在或不是目录：{root}",
                f"Error: path does not exist or is not a directory: {root}",
            ),
            file=sys.stderr,
        )
        return 2

    configure_git_env(
        proxy=args.proxy,
        use_shell_env=not args.no_shell_env,
        use_system_proxy=not args.no_system_proxy,
    )

    ignore_patterns = load_ignore_patterns(ignore_file)
    repos = discover_repositories(root, ignore_patterns)

    print_report_header(root)

    if not repos:
        print(ui("未发现可更新的 Git 仓库。", "No Git repositories found to update."))
        return 0

    summarizer: OpenAISummarizer | None = None
    for repo in repos:
        item = update_repository(repo, git_timeout=args.git_timeout)

        if item.has_update and not args.no_ai and summarizer is None:
            try:
                summarizer = OpenAISummarizer(api_timeout=args.api_timeout)
            except RuntimeError as exc:
                print(ui(f"错误：{exc}", f"Error: {exc}"), file=sys.stderr)
                return 2

        summarize_update(
            repo,
            item,
            summarizer=summarizer,
            git_timeout=args.git_timeout,
            no_ai=args.no_ai,
        )
        print_repo_result(item)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        print(ui("已收到 Ctrl+C，中断执行。", "Interrupted by Ctrl+C."), file=sys.stderr)
        raise SystemExit(130)
