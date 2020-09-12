#!/bin/bash

set -eo pipefail
ROOT=$(dirname "$0")
REPO_URL="https://github.com/Waino/LeBLEU.git"

# by default, the LEBLEU will be downloaded to ~/lmvr/lmvr-repo,
# and the virtual environment installed in ~/lmvr/lmvr-env.
# these can obviously be overridden by setting the environment vairables.
if [ -z "$LEBLEU_PATH" ] || [ -z "$LEBLEU_ENV_PATH" ]; then
    echo "Using default values..."
    source "$ROOT/lebleu-environment-variables.sh"
fi

# remove existing installations
rm -Rf "$LEBLEU_PATH" "$LEBLEU_ENV_PATH"

git clone "${REPO_URL}" "$LEBLEU_PATH"

# you can also customize the python 2.7 executable you give
# virtualenv. by default we use $(which python2), and assume
# the user normally runs python 3
if [ -z "$PYTHON2_EXECUTABLE" ]; then
    PYTHON2_EXECUTABLE=$(which python2)
fi

virtualenv "$LEBLEU_ENV_PATH" --python="$PYTHON2_EXECUTABLE"

# install
cd "$LEBLEU_PATH"
source "$LEBLEU_ENV_PATH/bin/activate"
python setup.py install
pip install numpy
git submodule init
git submodule update
cd python-Levenshtein
python setup.py install
deactivate
