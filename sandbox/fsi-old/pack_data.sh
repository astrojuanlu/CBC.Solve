#!/bin/sh

FILE=data.tar.gz

echo "Packing data to $FILE"
rm -f ~/$FILE
tar zcfv ~/$FILE `find results-*/* | grep -v bin | grep -v pvd | grep -v mesh`
ls -lh ~/$FILE
cat results-*/goal_functional_final.txt | wc
