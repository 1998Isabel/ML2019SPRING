#!/bin/sh
# $1 = input images $2 = outputs

# fgsm: resnet50
python fgsm_best.py --input $1 --output $2 --label labels.csv --part 0 
