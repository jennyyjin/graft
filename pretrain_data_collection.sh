#!/bin/bash

# Usage: bash run_captioning.sh /path/to/data_root

# Check if the data_path_root argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: bash $0 /path/to/data_root"
    exit 1
fi

DATA_PATH_ROOT=$1

# Run the Python script with the provided data_path_root
python3 batch_captioning.py --data_path_root "$DATA_PATH_ROOT"
