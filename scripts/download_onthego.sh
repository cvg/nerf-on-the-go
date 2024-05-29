#!/bin/bash

mkdir Datasets
wget https://cvg-data.inf.ethz.ch/on-the-go.zip
unzip on-the-go.zip -d Datasets
rm on-the-go.zip
# Base directory containing the sequence directories
base_dir="./Datasets/on-the-go"

# Loop through each sequence directory in the base directory
for seq_dir in "$base_dir"/*; do
    # Extract just the name of the directory (the sequence name)
    seq_name=$(basename "$seq_dir")
    echo "Processing sequence: $seq_name"

    # Determine the downsampling rate based on the sequence name
    if [ "$seq_name" = "arcdetriomphe" ] || [ "$seq_name" = "patio" ]; then
        rate=4
    else
        rate=8
    fi

    # Calculate percentage for resizing based on the downsample rate
    percentage=$(bc <<< "scale=2; 100 / $rate")

    # Directory names for images, defined relative to the base_dir
    original_images_dir="$seq_dir/images"
    downsampled_images_dir="$seq_dir/images_$rate"

    # Copy images to new directory before downsampling, handling both JPG and jpg
    cp -r "$original_images_dir" "$downsampled_images_dir"

    # Downsample images using mogrify for both JPG and jpg
    pushd "$downsampled_images_dir"
    ls | xargs -P 8 -I {} mogrify -resize ${percentage}% {}
    popd

done
