#!/bin/bash
# .claude/hooks/require-branch-on-edit.sh
# mainブランチでのファイル編集をブロック

INPUT=$(cat)

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [[ "$CURRENT_BRANCH" == "main" || "$CURRENT_BRANCH" == "master" ]]; then
  echo "BLOCKED: mainブランチで直接ファイルを編集できません。" >&2
  echo "先にブランチを作成してください: git checkout -b feature/xxx" >&2
  exit 2
fi

exit 0
