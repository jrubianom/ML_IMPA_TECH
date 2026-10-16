#!/usr/bin/env bash
# =====================================================================
# ML1 (IMPA) - Python environment for the lesson (lab computers)
#
# Usage: create a folder, go into it and run
#     bash setup_ubuntu.sh
#
# The folder then has an environment "env" with the lesson's packages,
# registered as a Jupyter kernel so VS Code can run the notebooks.
# No sudo needed: it uses the machine's Python (python3.10). Running it
# again reuses "env" and only installs what is missing.
# =====================================================================
set -euo pipefail

# Packages for the lesson (ISLP Chapter 4 lab)
PACKAGES="numpy pandas scipy matplotlib scikit-learn statsmodels ipykernel"

ENV_DIR="$PWD/env"
FOLDER="$(basename "$PWD")"
KERNEL="ml1-$(printf '%s' "$FOLDER" | tr -c 'A-Za-z0-9._-' '_')"

PY=""
for cmd in python3.10 python3; do
  if command -v "$cmd" >/dev/null 2>&1 &&
     "$cmd" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$cmd"
    break
  fi
done
[ -n "$PY" ] || { echo "ERROR: Python 3.10 or newer not found" >&2; exit 1; }

if ! "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> Creating $ENV_DIR ($("$PY" --version))"
  "$PY" -m venv "$ENV_DIR"
fi

echo "==> Installing the packages"
# shellcheck disable=SC2086
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check $PACKAGES
# ISLP (the book's package) without its dependencies: they include
# PyTorch (about 1 GB), which this lesson does not use.
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-deps ISLP

echo "==> Registering the Jupyter kernel 'ML1 ($FOLDER)'"
"$ENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL" \
  --display-name "ML1 ($FOLDER)"

"$ENV_DIR/bin/python" -c 'import numpy, pandas, scipy, matplotlib, sklearn, statsmodels, ISLP'
echo "==> Done. In VS Code, open this folder and pick the kernel 'env' or 'ML1 ($FOLDER)'."
