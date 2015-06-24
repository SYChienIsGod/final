# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:07:59 2015

@author: jan
"""

import util
import phonemes
for source in ('test','train','val'):    
    with open('ssvmdata/'+source+'.out','w') as f:
        ids = util.loadNpy('caffedata/'+source+'_ids.npy')
        yList=list()
        for line in file('ssvmdata/'+source+'.ssvm','rb'):
            ys = [int(s) for s in line.strip().split(' ')]
            yList+=ys
        for id_,label in zip(ids,yList):
            f.write('{} {}'.format(id_,phonemes.c48[label]))