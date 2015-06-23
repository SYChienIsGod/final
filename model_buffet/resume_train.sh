#!/bin/bash
~/caffe/build/tools/caffe train -snapshot ./snapshot_iter_5544000.solverstate -solver ./basenet_resume.solver -gpu 0 2>&1 | tee log_resume.txt
