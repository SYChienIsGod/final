#!/bin/bash
~/caffe/build/tools/caffe train -solver ./basenet.solver 2>&1 | tee log.txt
