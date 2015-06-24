#!/bin/bash -vx
./svm_empty_learn -c $1 ../ssvmdata/train.txt 	nn_svm_c$1.model
./svm_empty_classify ../ssvmdata/val.txt 		nn_svm_c$1.model ../ssvmdata/val.ssvm
./svm_empty_classify ../ssvmdata/train.txt 		nn_svm_c$1.model ../ssvmdata/train.ssvm
./svm_empty_classify ../ssvmdata/test.txt 		nn_svm_c$1.model ../ssvmdata/test.ssvm
cd .. 
python reshape_ssvm_out.py
