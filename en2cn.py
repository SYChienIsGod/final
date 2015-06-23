# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:22:39 2015

@author: jan
"""

import numpy as np

cnDictP='../data/conf/timit.chmap'

en = np.loadtxt(cnDictP,dtype='str_',usecols=(0,),delimiter='\t')
cn = np.loadtxt(cnDictP,dtype='str_',usecols=(1,),delimiter='\t')

en2cn=dict(zip(en,cn))

def translate(enWord):
    enWord=enWord.lower()
    if en2cn.has_key(enWord):
        return en2cn[enWord]
    else:
        raise Exception("English word {} not found in dictionary!".format(enWord))
        
def translateSentence(enSent):
    r = ''
    for w in enSent:
        r = r+translate(w)