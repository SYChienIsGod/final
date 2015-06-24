#!/bin/bash
cd ..
python write_feat.py
cd caffedata
~/Dev/Repo/caffe/build/tools/caffe train -solver ./convnet.solver
