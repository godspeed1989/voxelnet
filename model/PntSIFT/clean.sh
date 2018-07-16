#!/bin/bash

cd ./grouping_op
make clean
cd ..

cd ./pointSIFT_op
make clean
cd ..

rm -rf ./grouping_op/__pycache__
rm -rf ./pointSIFT_op/__pycache__
rm -rf ./__pycache__
