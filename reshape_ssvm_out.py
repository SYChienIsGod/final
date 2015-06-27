# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:07:59 2015

@author: jan
"""

import util
import phonemes
import numpy as np
for source in ('test','train','val'):    
    with open('ssvmdata/'+source+'.out','w') as f:
        ids = util.loadNpy('caffedata/'+source+'_ids.npy')
        yList=list()
        for line in file('ssvmdata/'+source+'.ssvm','rb'):
            ys = [int(s) for s in line.strip().split(' ')]
            yList+=ys
        for id_,label in zip(ids.tolist(),yList):
            f.write('{} {} \n'.format(id_,phonemes.c39[label]))
        if source == 'val':
            gold=util.loadNpy('caffedata/val_labels.npy')
            nCorrect=np.count_nonzero(yList==gold)
            print 'Validation accuracy is now {}%'.format(float(nCorrect)/float(gold.shape[0])*100.0)