# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:13:22 2015

@author: jan
"""

import numpy


phonMapP='../data/conf/phones.60-48-39.map'

c60=dict()
c48=dict()
c39=dict()

ph60 = dict()
ph48=dict()
ph39=dict()
ph60_ph48=dict()


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
        if len(data) > 2 and not ph39.has_key(data[2]):
            ph39[data[2]] = len(ph39)
            c39[ph39[data[2]]]=data[2]

def trans_ph60_ph48(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_ph60_ph48(w))
        return r
    else:
        return ph60_ph48[d]
            
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

def trans_c48(d):
    if type(d) == list:
        r = list()
        for w in d:
            r.append(trans_c48(w))
        return r
    else:
        return c48[d]

