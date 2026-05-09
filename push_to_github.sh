#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# HAQOA — One-shot GitHub push script
# Usage: bash push_to_github.sh <your-github-username> <repo-name>
# Example: bash push_to_github.sh johndoe HAQOA
# Requires: git, curl, a Personal Access Token with repo scope
# ──────────────────────────────────────────────────────────────

set -e

USERNAME=${1:?'Usage: bash push_to_github.sh <github-username> <repo-name>'}
REPO=${2:-"HAQOA"}

echo "Enter your GitHub Personal Access Token (repo scope):"
read -rs TOKEN
echo ""

echo ">>> Creating repository '$REPO' on GitHub..."
HTTP_CODE=$(curl -s -o /tmp/gh_response.json -w "%{http_code}" \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  -d "{\"name\":\"$REPO\",\"description\":\"HAQOA — Hybrid AI-Assisted Quantum-Inspired Optimization Architecture\",\"private\":false}" \
  https://api.github.com/user/repos)

if [ "$HTTP_CODE" == "201" ]; then
  echo ">>> Repository created successfully."
elif [ "$HTTP_CODE" == "422" ]; then
  echo ">>> Repository already exists — pushing to existing repo."
else
  echo ">>> GitHub API error (HTTP $HTTP_CODE):"
  cat /tmp/gh_response.json
  exit 1
fi

REMOTE="https://${USERNAME}:${TOKEN}@github.com/${USERNAME}/${REPO}.git"

git init
git add .
git commit -m "feat: HAQOA Phase 1 + Phase 2 — AQSE-v1 engine, TSP benchmark, all baselines"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin main --force

echo ""
echo "✅  Done! Repository live at: https://github.com/$USERNAME/$REPO"
