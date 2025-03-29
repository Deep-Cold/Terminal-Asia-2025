#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/../venv/bin/activate"
PYTHON_EXE=${PYTHON_CMD:-python3}
$PYTHON_EXE -u "$DIR/algo_strategy_ddpg.py"
