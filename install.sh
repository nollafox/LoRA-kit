#!/usr/bin/env sh
set -e

python -m pip install .

if command -v lorakit >/dev/null 2>&1; then
  lorakit install --with-models
else
  python -m lorakit.cli install --with-models
fi
