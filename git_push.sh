#!/bin/sh

echo "COMMIT MSG: $1"

git add .
git commit -m "$1"
git push origin master
