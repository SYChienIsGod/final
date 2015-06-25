# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:13:22 2015

@author: jan
"""

import numpy


phonMapP='../data/conf/phones.60-48-39.map'

# number to phoneme
c60=dict()
c48=dict()
c39=dict()

# phoneme to number
ph60 = dict()
ph48=dict()
ph39=dict()
ph60_ph48=dict()
ph60_ph39=dict()
ph48_ph39=dict()


with open(phonMapP) as f:
    for l in f:
        data=l.strip().split('\t');
        if not ph60.has_key(data[0]):
            ph60[data[0]] = len(ph60)
            c60[ph60[data[0]]]=data[0]
        if len(data) > 1 :
            if not ph48.has_key(data[1]):
                ph48[data[1]] = len(ph48)
                c48[ph48[data[1]]]=data[1]
            ph60_ph48[data[0]]=data[1]
        if len(data) > 2 :
            if not ph39.has_key(data[2]):
                ph39[data[2]] = len(ph39)
                c39[ph39[data[2]]]=data[2]
            ph60_ph39[data[0]]=data[2]
            ph48_ph39[data[1]]=data[2]

def trans_ph60_ph48(d):
    if type(d) == list:
        for w in d:
            r.append(trans_ph60_ph48(w))
        return r
    elif type(d) == numpy.ndarray:
        r = numpy.zeros(shape=(d.shape),dtype='|S5')
        for i in range(d.size):
            r[i]=trans_ph60_ph48(d[i])
        return r 
    else:
        return ph60_ph48[d]

def trans_ph60_ph39(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_ph60_ph39(w))
        return r
    elif type(d) == numpy.ndarray:
        r = numpy.zeros(shape=(d.shape),dtype='|S5')
        for i in range(d.size):
            r[i]=trans_ph60_ph39(d[i])
        return r 
    else:
        return ph60_ph39[d]

def trans_ph48_ph39(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_ph48_ph39(w))
        return r
    elif type(d) == numpy.ndarray:
        r = numpy.zeros(shape=(d.shape),dtype='|S5')
        for i in range(d.size):
            r[i]=trans_ph48_ph39(d[i])
        return r  
    else:
        return ph48_ph39[d]

def trans_ph48(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_ph48(w))
        return r
    elif type(d) == numpy.ndarray:
        r = numpy.zeros(shape=(d.shape),dtype='int')
        for i in range(d.size):
            r[i]=trans_ph48(d[i])
        return r
    else:
        return ph48[d]

def trans_ph39(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_ph39(w))
        return r
    elif type(d) == numpy.ndarray:
        r = numpy.zeros(shape=(d.shape),dtype='int')
        for i in range(d.size):
            r[i]=trans_ph39(d[i])
        return r
    else:
        return ph39[d]

def trans_c48(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_c48(w))
        return r
    else:
        return c48[d]

def trans_c39(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_c39(w))
        return r
    else:
        return c39[d]

