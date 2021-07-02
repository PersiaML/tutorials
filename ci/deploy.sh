#!/usr/bin/env bash
set -euo pipefail

mv -f book/html/* .

git init
git config user.name "PersiaML GitHub Action"
git config user.email "no-reply@fake.com"

git remote add upstream "https://$GH_TOKEN@github.com/PersiaML/tutorials.git"
git fetch upstream
git reset upstream/gh-pages

touch .

git add -A .
git commit -m "rebuild pages at ${rev}"
git push -q upstream HEAD:gh-pages > /dev/null 2>&1
