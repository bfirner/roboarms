#! /usr/bin/bash

if [ $# -ne 1 ]; then
    echo "This script requires the name of the webdataset file whose contents should be shuffled."
    exit
fi

# Mount the tar archive with archivemount
mountfile=mnt.$$
mkdir $mountfile
archivemount $1 $mountfile

if [ $? -ne 0 ]; then
    echo "Failed to mount the archive, aborting."
    rm $mountfile
    exit
fi

# Find the numbers of records
cd $mountfile

# Get the list of numbers of the samples in the archive
numbers=$(ls | sed 's/\..*//g' | sed 's/.*_//g' | sort -n | uniq)
number_array=($numbers)
num_numbers=${#number_array[@]}
num_idx=0

# Loop through the original names and add the shuffled prefixes to alter their orders
# Make a list of new target names in randomized order
for dst in $(seq 0 $((num_numbers-1)) | shuf); do

    src=${number_array[$num_idx]}
    num_idx=$((num_idx+1))

    #echo "Source index is ${src}"

    for file in $(ls *_${src}.*); do
        #echo "moving $file to ${dst}_$file"
        mv "$file" "${dst}_$file"
        # Rename the files. Originally the plan was to swap like this:
        #swapfile=tmp.$$

        ## This trick from https://stackoverflow.com/questions/1115904/shortest-way-to-swap-two-files-in-bash
        ## I believe that the link will not create a copy of the file
        #ln "$src" "$swapfile";
        #mv -f "$dst" "$src";
        #mv "$swapfile" "$dst"

    done
done

## Clean up
cd ../
umount $mountfile
rm -r $mountfile
