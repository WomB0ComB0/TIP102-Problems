#!/bin/bash

for dir in $(find . -type d -name "unit_*"); do
    rm -rf $dir
done

rm -rf __pycache__
