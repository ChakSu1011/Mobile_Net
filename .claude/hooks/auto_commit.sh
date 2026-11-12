#!/usr/bin/env bash
# 模块：Claude Code 自动提交钩子
#
# 职责：
#   在 Stop / SessionEnd 事件中提交工作区变更，并推送到当前分支的 origin；
#   没有变更时不做任何事。
#
# 边界：
#   - 不执行测试、不判断代码质量；
#   - 不改写历史、不使用 --force；
#   - 检测到疑似敏感文件时拒绝提交；
#   - 推送失败时保留本地提交，不阻断 Claude Code。
#
# 输入：
#   AUTO_COMMIT_DRY_RUN=1 或 --dry-run：只打印将提交的文件，不提交/推送。
#   AUTO_COMMIT_MESSAGE：自定义提交标题；默认 chore: auto-commit <时间>。
#
# 输出：
#   标准输出为 JSON systemMessage；退出码始终为 0。
#
# 网络：
#   脚本不修改网络配置。推送失败时先检查 .git/config 中的
#   http.https://github.com.proxy 或系统代理是否可用；失败只保留本地提交。

set -u

dry_run=0
if [ "${AUTO_COMMIT_DRY_RUN:-0}" = "1" ] || [ "${1:-}" = "--dry-run" ]; then
  dry_run=1
fi

# 定位仓库根目录；Git Bash 可以直接 cd Windows 反斜杠路径。
repo_root="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$repo_root" ]; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    printf '%s\n' \
      '{"systemMessage":"auto-commit: not inside a git repository"}'
    exit 0
  }
fi
cd "$repo_root" || {
  printf '%s\n' \
    '{"systemMessage":"auto-commit: cannot enter repository root"}'
  exit 0
}

# 防止 Stop 与 SessionEnd 并发提交；worktree 下 .git 是文件，
# 必须用 git rev-parse --git-path 定位锁目录。
lock_dir="$(git rev-parse --git-path claude-auto-commit.lock 2>/dev/null)"
if [ -z "$lock_dir" ] || ! mkdir "$lock_dir" 2>/dev/null; then
  exit 0
fi
cleanup() { rmdir "$lock_dir" 2>/dev/null || true; }
trap cleanup EXIT

# 收集未暂存、已暂存、未跟踪的变更（尊重 .gitignore）。
changed_files="$(
  {
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | sort -u
)"

if [ -z "$changed_files" ]; then
  exit 0
fi

# 敏感文件拦截。模板和公钥放行，私钥、证书、凭据文件拒绝提交。
suspicious="$(
  printf '%s\n' "$changed_files" \
    | grep -Ev '(^|/)\.env\.(example|sample|template)$' \
    | grep -E '(^|/)(\.env(\..*)?|\.npmrc|\.pypirc|\.netrc|credentials(\..*)?|secrets?(\..*)?|id_(rsa|dsa|ecdsa|ed25519)|.*\.(pem|key|p12|pfx|jks|keystore|ovpn))$' \
    || true
)"
if [ -n "$suspicious" ]; then
  message="auto-commit blocked: suspicious file(s): $(printf '%s' "$suspicious" | tr '\n' ' ')"
  escaped="$(printf '%s' "$message" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  printf '{"systemMessage":"%s"}\n' "$escaped"
  exit 0
fi

if [ "$dry_run" = "1" ]; then
  message="auto-commit dry-run: would commit: $(printf '%s' "$changed_files" | tr '\n' ' ')"
  escaped="$(printf '%s' "$message" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  printf '{"systemMessage":"%s"}\n' "$escaped"
  exit 0
fi

git add -A
if git diff --cached --quiet; then
  exit 0
fi

timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
commit_message="${AUTO_COMMIT_MESSAGE:-chore: auto-commit $timestamp}"
summary="$(git diff --cached --stat | tail -n 1)"
if ! git commit -m "$commit_message" \
    -m "$summary" \
    -m "Co-Authored-By: Claude Code <noreply@anthropic.com>" >/dev/null 2>&1; then
  printf '%s\n' \
    '{"systemMessage":"auto-commit: git commit failed; see git status"}'
  exit 0
fi

branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
if [ -z "$branch" ]; then
  printf '%s\n' \
    '{"systemMessage":"auto-commit: detached HEAD; commit kept local"}'
  exit 0
fi

# 禁止交互式凭据提示，避免钩子挂起；推送失败时保留本地提交。
if ! GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=Never \
    git push origin "$branch" >/dev/null 2>&1; then
  printf '%s\n' \
    '{"systemMessage":"auto-commit: commit created locally, but push failed; check credentials/network"}'
  exit 0
fi

printf '{"systemMessage":"auto-commit: committed and pushed to origin/%s"}\n' \
  "$branch"
