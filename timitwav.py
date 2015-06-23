# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:43:57 2015

@author: jan
"""

import scipy.io.wavfile as wav
import numpy as np
import paths
from features import mfcc
from features import logfbank
from features import fbank
from features import ssc
from features import framesig
import util
import speakersent
wavFolder = '../data/wav/'

fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
fbank_test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))

sswitr = speakersent.getSpeakerSentenceWordId(fbank_train_ids)
sswite = speakersent.getSpeakerSentenceWordId(fbank_test_ids)
#%%

#speakerSets=speakersent.getSpeakerSets(fbank_train_ids)
#sentGroupId,sentGroupCnt=speakersent.getSpeakerSentenceGroups(fbank_train_ids,speakerSets)
#validationSet = speakersent.getValidationSet(fbank_train_ids,sentGroupId,sentGroupCnt)
#trainingSet = set(sentGroupId).difference(validationSet)
#%%
winlen = 0.025
winstep = 0.01
def toWavFileName(speaker,sent):
    return wavFolder+speaker+'_'+sent+'.wav'

def getWavFile(speaker,sent):
    rate,sig=wav.read(toWavFileName(speaker,sent))
    return rate,sig

def getFrames(speaker,sent):
    r,sig = getWavFile(speaker,sent)
    frames = framesig(sig,winlen*r,winstep*r)
    frames = frames[:-1] # Inclomplete frames at the end are cut off
    return frames

def computeDelta(frames,N=2):
    tmp_frames = np.pad(frames,((N,N),(0,0)),mode='edge')
    delta = np.zeros_like(frames)
    norm = N*(N+1)*(2*N+1)/6.0*2.0
    for i in range(delta.shape[0]):        
        for k in range(1,N+1):
            delta[i,:] += k*(-tmp_frames[i-k+N,:]+tmp_frames[i+k+N,:])/norm
    return delta            
#%%
    
#speaker_='faem0'
#sent_='si1392'
#
#r,sig = getWavFile(speaker_,sent_)
#fbanks,energy= fbank(sig,r,winlen=winlen,winstep=winstep,nfilt=69)
#delta=computeDelta(fbanks,N=1)
#print fbanks[0]
#print fbanks[1]
#print delta[0]
            
#%%
def computeFBANKDeltaDelta(sswi,NFilt=40,NDelta=2):
    nframes = speakersent.countFrames(sswi)
    NFilt=NFilt+1 # Energy values count as one extra
    features = np.zeros(shape=(nframes,NFilt*3))
    util.startprogress("FBANK Features")
    frameCnt=0
    for k,v in sswi.iteritems(): # k: speakerId, v: dict with sentenceId
        speaker_=k
        for k2,v2 in v.iteritems(): # k2: sentenceId, v: dict with frameId -> entry
            sent_=k2
            r,sig=getWavFile(speaker_,sent_)
            fbank_frames,energy=fbank(sig,r,winlen=winlen,winstep=winstep,nfilt=NFilt-1)
            fbank_frames=np.log(np.append(np.reshape(energy,(energy.shape[0],1)),fbank_frames,axis=1))
            delta_1 = computeDelta(fbank_frames,N=NDelta)
            delta_2 = computeDelta(delta_1,N=NDelta)
            util.progress(float(frameCnt)/nframes*100.0)
            for i in range(len(v2)):
                frameCnt+=1
                features[v2[i+1],:NFilt]=fbank_frames[i,:]                
                features[v2[i+1],NFilt:2*NFilt]=delta_1[i,:]                
                features[v2[i+1],2*NFilt:]=delta_2[i,:]
    util.endprogress()
    return features

trainingFeatures = computeFBANKDeltaDelta(sswitr)
np.save('../data/fbank/train_delta2',trainingFeatures)
testFeatures = computeFBANKDeltaDelta(sswite)
np.save('../data/fbank/test_delta2',testFeatures)