#!/bin/bash

first=true
for f in `find $1 -name nodes.csv`; do
    if [ "$first" = true ]; then
        head -n 1 $f > nodes.csv
        first=false
    fi
    tail -n +2 $f >> nodes.csv
done

first=true
for f in `find $1 -name edges.csv`; do
    if [ "$first" = true ]; then
        head -n 1 $f > edges.csv
        first=false
    fi
    tail -n +2 $f >> edges.csv
done
