#!/usr/bin/env bash

#
# Generate the lattices for the whole database.
#

if [ $# -ne 3 ]; then
    echo "usage: $0 <latt_conf_dir> <posts_dir> <out_dir>"
    exit 1
fi

latt_conf_dir=$1
post_dir=$2
out_dir=$3

# Create the output _directory.
mkdir -p "$out_dir"

for fname in $(find "$post_dir" -name "*.posts") ; do
    echo "$fname"
    steps/create_lattice.sh "$latt_conf_dir" "$fname" "$out_dir" > /dev/null || exit 1
done

date > "$out_dir"/.done

