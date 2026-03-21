#!/bin/bash
# .claude/hooks/require-branch-on-commit.sh
# mainブランチへの直接コミットをブロックする

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# git commit コマンドでなければスルー
echo "$COMMAND" | grep -q "git commit" || exit 0

# 現在のブランチを確認
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

# main / master への直接コミットをブロック
if [[ "$CURRENT_BRANCH" == "main" || "$CURRENT_BRANCH" == "master" ]]; then
  echo "BLOCKED: 現在 '$CURRENT_BRANCH' ブランチにいます。" >&2
  echo "作業用ブランチを切ってからコミットしてください。" >&2
  echo "例: git checkout -b feature/your-feature-name" >&2
  exit 2
fi

exit 0
