#!/usr/bin/env bash
# =====================================================================
# ML1 (IMPA) - Python environment for the lesson - macOS
#
# Usage: create a folder, go into it and run (in the Terminal app)
#     bash setup_mac.sh
#
# The folder then has an environment "env" with the lesson's packages,
# registered as a Jupyter kernel so VS Code can run the notebooks.
# It uses the Python already installed (3.10 or newer). Running it
# again reuses "env" and only installs what is missing.
# =====================================================================
set -euo pipefail

# Packages for the lesson (ISLP Chapter 4 lab)
PACKAGES="numpy pandas scipy matplotlib scikit-learn statsmodels ipykernel"

ENV_DIR="$PWD/env"
FOLDER="$(basename "$PWD")"
KERNEL="ml1-$(printf '%s' "$FOLDER" | tr -c 'A-Za-z0-9._-' '_')"

PY=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3; do
  p="$(command -v "$cmd" || true)"
  [ -n "$p" ] || continue
  # Apple's /usr/bin/python3 opens an install dialog when the Command
  # Line Tools are missing (and it is too old anyway)
  if [ "$p" = /usr/bin/python3 ] && ! xcode-select -p >/dev/null 2>&1; then
    continue
  fi
  if "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$p"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python 3.10 or newer not found. Install it from" >&2
  echo "https://www.python.org/downloads/macos/ and run this script again." >&2
  exit 1
fi

if ! "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> Creating $ENV_DIR ($("$PY" --version))"
  "$PY" -m venv "$ENV_DIR"
fi

echo "==> Installing the packages"
# --prefer-binary: if the newest version of a package has no ready-made
# build for this Mac, take the newest one that has (instead of compiling)
# shellcheck disable=SC2086
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  --prefer-binary $PACKAGES
# ISLP (the book's package) without its dependencies: they include
# PyTorch (about 1 GB), which this lesson does not use.
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-deps ISLP

echo "==> Registering the Jupyter kernel 'ML1 ($FOLDER)'"
"$ENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL" \
  --display-name "ML1 ($FOLDER)"

"$ENV_DIR/bin/python" -c 'import numpy, pandas, scipy, matplotlib, sklearn, statsmodels, ISLP'
echo "==> Done. In VS Code, open this folder and pick the kernel 'env' or 'ML1 ($FOLDER)'."
