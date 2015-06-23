
#for CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


#for Torch7
export PATH=/home/baconx2/torch/install/bin:$PATH  
export LD_LIBRARY_PATH=/home/baconx2/torch/install/lib:$LD_LIBRARY_PATH  
export DYLD_LIBRARY_PATH=/home/baconx2/torch/install/lib:$DYLD_LIBRARY_PATH  

#for cudnn
export CPATH=/opt/cudnn-6.5-linux-x64-v2:$CPATH
export LIBRARY_PATH=/opt/cudnn-6.5-linux-x64-v2:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH


export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export PYTHONPATH=/home/baconx2/caffe/python:/usr/include/google/protobuf:/home/baconx2/LPIRC/BING-Objectness/build

export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH

export PATH=/opt/MATLAB/R2013a/bin:$PATH

