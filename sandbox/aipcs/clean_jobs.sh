#!/bin/sh

for f in `qstat | grep $USER | cut -d' ' -f1`; do
    echo "Deleting $f"
    qdel $f
done
