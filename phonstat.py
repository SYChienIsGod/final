# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:10:50 2015

@author: jan
"""

import numpy as np
import paths
import timitdict
import speakersent
import phonemes
fbank_train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
fbank_train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))

#%%

def encodeUtterance(labels):
    last = ''
    cnt=0
    result=list()
    for i in range(len(labels)):
        if labels[i] != last:
            if cnt>0:
                result.append(last)
            cnt=1
            last=labels[i]
        else:
            cnt+=1
    result.append(last)
    return phonemes.trans_ph39(phonemes.trans_ph48_ph39(result))
    

sswi = speakersent.getSpeakerSentenceWordId(fbank_train_lab_ids)

def getSentenceData(sentenceFile, labels, sswi):
    sentences=dict()
    with open(sentenceFile) as f:
        for l in f:
            data=l.strip().split(',',1)
            id_=data[0]
            sent=data[1]
            d=id_.split('_')
            speaker=d[0]
            sentence=d[1]
            if not sentences.has_key(speaker):
                sentences[speaker]=dict()
            utt=[labels[sswi[speaker][sentence][k+1]] for k in range(len(sswi[speaker][sentence]))]
            sentences[speaker][sentence]={'timit':timitdict.translateSentence(sent),
                'timitc':timitdict.translateSentenceC(sent),'utt':phonemes.trans_ph39(phonemes.trans_ph48_ph39(utt)),'orig':sent,'uttEnc':encodeUtterance(utt)}
    return sentences

sentences=getSentenceData('../data/sentence/train.set',fbank_train_lab,sswi)

#%%
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
def getAlignment(timit, utterance):
    tim = list()
    for li in timit:
        for ph in li:
            tim.append(ph)
    a=Sequence(tim)
    b=Sequence(utterance)
    v=Vocabulary()
    aEnc=v.encodeSequence(a)
    bEnc=v.encodeSequence(b)
    scoring=SimpleScoring(2,-1)
    aligner=GlobalSequenceAligner(scoring,-2)
    score,encodeds= aligner.align(aEnc,bEnc,backtrace=True)
    for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        return alignment

#%%
    
for speaker, sents in sentences.iteritems():
    for sent, data in sents.iteritems():
        sentences[speaker][sent]['utt2tim'] = getAlignment(data['timitc'],data['uttEnc'])

#%%

# Analyse Single to Single replacements

replacements=np.zeros(shape=(39,39))

for speakers, sents in sentences.iteritems():
    for sent, data in sents.iteritems():
        for i in range(len(data['utt2tim'])):
            tim = data['utt2tim'][i][0]
            utt = data['utt2tim'][i][1]
            if tim!='-' and utt!='-':                
                replacements[tim,utt]+=1
#%%
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
#    tick_marks = np.arange(39)
#    plt.xticks(tick_marks, , rotation=45)
#    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plt.figure()
plot_confusion_matrix(replacements/replacements.sum(axis=1))

#%%

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

rep=replacements/replacements.sum(axis=1)
rep[rep==np.inf]=0
for i in range(39):
    rep[i,i]=0.0
idx = np.where(rep>0.05)
timitId = idx[0]
uttId = idx[1]
timit2utt=dict()
for i in range(39):
    timit2utt[i]=list()
    timit2utt[i].append(i)
for i in range(len(idx)):
    timit2utt[timitId[i]].append(uttId[i])

stopCount = 0
with open('../data/conf/timitdic_mod-005.txt','w') as f:
    for word, phons in timitdict.timitDictC.iteritems():
        counts = np.ones(shape=(len(phons),))
        arr=list()
        for i in range(len(phons)):
            arr.append(np.arange(len(timit2utt[phons[i]])))
        mutations=cartesian(arr)
        for i in range(mutations.shape[0]):
             f.write(word+('_'*i)+'  /'+' '.join([phonemes.c39[timit2utt[phon][mut]] for phon,mut in zip(phons,mutations[i])])+'/\n')
        stopCount+=1
#        if stopCount > 2:
#            break

#%%
phonLen=dict()
last = ''
cnt=0
for i in range(fbank_train_lab.size):
    if fbank_train_lab[i]!=last:
        if cnt>0:
            #print '{}{}'.format(last,cnt),
            if not phonLen.has_key(last):
                phonLen[last]=list()
            phonLen[last].append(cnt)
        cnt=1
        last=fbank_train_lab[i]
    else:
        cnt+=1
#print '{}{}'.format(last,cnt)
phonLen[last].append(cnt)

for k,v in phonLen.iteritems():
    r=np.array(v)
    print 'Phoneme {}: mean={:0.1f} var={:0.2f} median={} int=[{},{}]'.format(k,np.mean(r),np.var(r),np.median(r),np.min(r),np.max(r))