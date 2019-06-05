#!/bin/sh
# bash  hw8_test.sh  <testing data>  <prediction file>
# $1 = <testing data> $2 = <prediction file>

# test 
python src/test_quantize.py --data_path $1 --weights model/weights --output_path $2