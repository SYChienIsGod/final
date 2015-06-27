# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:36:44 2015

@author: jan
"""

import re

timitDictP = '../data/conf/timitdic.txt'

timitDict = dict()

with open(timitDictP) as f:
    for line in f:
        data=line.strip().split(' ')
        if data[0]==';':
            continue
        tmp=list()
        d=data[0].split('~',1)
        for i in range(1,len(data)):
            w = data[i].replace('/','')
            w = w.replace('1','')
            w = w.replace('2','')
            w = w.strip()
            if len(w) > 0:
                tmp.append(w)
        timitDict[d[0]]=tmp

phons = set()
for v in timitDict.itervalues():
    phons = phons.union(set(v))
    
import phonemes

timitDictC = dict()

for k,v in timitDict.iteritems():
    timitDictC[k]=phonemes.trans_ph39(phonemes.trans_ph60_ph39(v))

def translateSentence(sent):
    if type(sent)==str:        
        sent=sent.strip()
        if sent[-1]=='.':
            sent=sent[:-1]
        sent=sent.split(' ')
    tmp=list()
    for w in sent:
        w=re.sub(r'[,\?;:"!]','',w.lower().strip().replace('--',''))
        if len(w)==0:
            continue
        tmp.append(timitDict[w])
    return tmp

def translateSentenceC(sent):
    if type(sent)==str:        
        sent=sent.strip()
        if sent[-1]=='.':
            sent=sent[:-1]
        sent=sent.split(' ')
    tmp=list()
    for w in sent:
        w=re.sub(r'[,\?;:"!]','',w.lower().strip().replace('--',''))
        if len(w)==0:
            continue
        tmp.append(timitDictC[w])
    return tmp