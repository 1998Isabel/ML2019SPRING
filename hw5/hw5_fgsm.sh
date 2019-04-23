#!/bin/sh
# $1 = input images $2 = outputs

# fgsm: densenet121
python fgsm.py --input $1 --output $2 --label labels.csv --part 0 
