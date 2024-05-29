#!/bin/bash

# Base directory containing the sequence directories
base_dir="./Datasets/on-the-go"

# Define the sequences that need special parameters
special_seqs=("arcdetriomphe" "patio")

# Loop through each sequence directory in the base directory
for seq_dir in "$base_dir"/*; do
    # Extract just the name of the directory (the sequence name)
    seq_name=$(basename "$seq_dir")

    # Check if the sequence is one of the special cases
    if [[ " ${special_seqs[@]} " =~ " $seq_name " ]]; then
        # Run feature extraction with additional parameters for special sequences
        python scripts/feature_extract.py --seq "$seq_name" --H 1080 --W 1920 --rate 2
    else
        # Run feature extraction without additional parameters for all other sequences
        python scripts/feature_extract.py --seq "$seq_name"
    fi
done
