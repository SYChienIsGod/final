import sys
import csv,os
import numpy as np
import time
sys.path.append('/opt/caffe_cudnn2/python/caffe') # MediaG caffe path 
caffe_root = '/opt/caffe_cudnn2/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import paths

predLayerName = 'out' #Output of the last layer

# Load dictionary 
import phonemes

# Load sequence name
fbank_test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))

# Put model you want to blend here (Test file is specify in basenet_test.prototxt)
# Net1
description_1 = './caffedata/basenet_test.prototxt'
learnedModel_1 = './caffedata/snapshot_iter_6930000.caffemodel'
# Net2
description_2 = './caffedata/basenet_test1.prototxt'   # You need to duplicate test.lvl for each model
learnedModel_2 = './caffedata1/snapshot_iter_6930000.caffemodel'
# Net3
# ...

# Submission file
submissionName = 'submission_smooth.txt'

if __name__ == "__main__":  
  fout = open(submissionName, 'w');
  w = csv.writer(fout);
  
  caffe.set_mode_gpu()
  net_1 = caffe.Net(description_1, learnedModel_1, caffe.TEST)
  net_2 = caffe.Net(description_2, learnedModel_2, caffe.TEST)  
  #net_3 = 
  #net_4 =   

  read = 0
  buff1 = 0
  buff2 = 0
  buff3 = 0
  while(read <= 166114):
    start = time.time()
    res_1 = net_1.forward() # this will load the next mini-batch as defined in the net (rewinds)
    res_2 = net_2.forward()
    print ("Time for getting the batch " + str((time.time() - start)) + " " + str(read))
    preds = np.add(net_1.blobs[predLayerName].data,net_2.blobs[predLayerName].data) 
    pred = preds.argmax()
    # Smooth    
    if(read>=3):
      if(pred==buff3 and buff2!=buff1): # Ex. 1 1 1 1 2 3 4 4  => 1 1 1 1 1 4 4 4
        buff1 = pred
        buff2 = buff3       
      if(buff2==pred): # Ex. 1 1 1 2 1 1 2 2 => 1 1 1 1 1 1 2 2 
        buff1 = pred
      fout.write(fbank_test_ids[read-2] + ' ' + phonemes.c48[buff2] +'\n')    
    elif(read>1):
      fout.write(fbank_test_ids[read-2] + ' ' + phonemes.c48[pred] +'\n') 
    buff3=buff2
    buff2=buff1
    buff1=pred
    read += 1










