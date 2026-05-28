## Multi-Repository Git Update Summarizer

`update_repos.py` updates all top-level Git repositories under the current directory and prints a concise Markdown report as it works. Directories listed in `git_ignore.txt` are skipped.

For each repository, the script records the current commit, fetches the upstream branch, performs a fast-forward update when possible, and immediately prints the result. If new commits were pulled, it collects the commit log and diff statistics, then uses an OpenAI-compatible chat model to summarize implementation-relevant changes while ignoring CI/CD-only, documentation-only, formatting-only, version bump, and lockfile-only updates.

### Configuration

```bash
export OPENAI_API_BASE="https://your-openai-compatible-endpoint"
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-model-name"
```
Note: If you use DeepSeek's services, you don't need to set OPENAI_API_BASE and OPENAI_API_MODEL.
