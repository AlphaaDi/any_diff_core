#!/bin/bash

# Loop through each subdirectory in the current directory
for dir in */ ; do
    # Check if 'setup.py' exists in the directory
    if [[ -f "$dir/setup.py" ]]; then
        echo "Building package in $dir"
        # Navigate into the directory
        cd "$dir"
        # Run the build command (e.g., 'python setup.py sdist bdist_wheel')
        pip install -e .
        # Navigate back to the parent directory
        cd ..
    else
        echo "No setup.py found in $dir, skipping"
    fi
done