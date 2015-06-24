# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:20:45 2015

@author: jan
"""

import leveldb
import caffe_pb2
import numpy as np

def writeData(data,labels,path,order=None):
    db = leveldb.LevelDB(path)
    batch = leveldb.WriteBatch()
    if type(order)==type(None):
        perm=np.arange(labels.size)
    else:
        perm=order
    for i in range(data.shape[0]):
        datum = caffe_pb2.Datum()
        datum.channels=1
        datum.height=1
        datum.width=data.shape[1]
        datum.float_data.extend(list(data[perm[i]].astype(float)))
        datum.label=labels[perm[i]]
        result = datum.SerializeToString()
        key = i
        batch.Put(str(key).zfill(6),result)
    db.Write(batch,sync=True)
    
def writeConvData(data,labels,path,order=None):
    db = leveldb.LevelDB(path)
    batch = leveldb.WriteBatch()
    if type(order)==type(None):
        perm=np.arange(labels.size)
    else:
        perm=order
    for i in range(data.shape[0]):
        datum = caffe_pb2.Datum()
        datum.channels=1
        datum.height=data.shape[1]
        datum.width=data.shape[2]
        datum.float_data.extend(list(np.ravel(data[perm[i]]).astype(float)))
        datum.label=labels[perm[i]]
        result = datum.SerializeToString()
        key = i
        batch.Put(str(key).zfill(6),result) #zfill(6), use iterator to output data inorder
    db.Write(batch,sync=True)
