# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:05:06 2015

@author: jan
"""

import paths
import numpy
import networkx as nx
#import cPickle

#trainIds = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',usecols=(0,))

#%%
#speakerSentences = [(sentence.split('_')[0],sentence.split('_')[1]) for sentence in trainIds]
#
#G = nx.Graph()
#
#for speaker, sentence in speakerSentences:
#    G.add_edge(speaker,sentence)
#    
#subsets = list(nx.connected_components(G))

#%%

#speakerSets = list()
#
#for subset in subsets:
#    speakerSet = list()
#    for elem in subset:
#        if elem[0] in ('f','m'):
#            speakerSet.append(elem)
#    speakerSets.append(speakerSet)
    
#%%



def getSpeakerSentenceGroups(ids):
    speakerSentences = [(sentence.split('_')[0],sentence.split('_')[1]) for sentence in ids]
    G = nx.Graph()    
    for speaker, sentence in speakerSentences:
        G.add_edge(speaker,sentence)        
    subsets = list(nx.connected_components(G))
    speakerSets = list()
    for subset in subsets:
        speakerSet = list()
        for elem in subset:
            if elem[0] in ('f','m'):
                speakerSet.append(elem)
        speakerSets.append(speakerSet)
        
    sentenceGroupId = list()
    sentenceGroupInst = [0] * len(speakerSets)
    for speaker, sentence in speakerSentences:
        for i in range(len(speakerSets)):
            if speaker in speakerSets[i]:
                sentenceGroupId.append(i)
                sentenceGroupInst[i] += 1
    return sentenceGroupId,sentenceGroupInst
            
#%% Save the information
            
#f = file(paths.pathToSentenceGroupIds,'wb')
#cPickle.dump(sentenceGroupId,f)
#f.close()

# Return the SpeakerId, the Sentence Id and the WordId            
def SpSeWo(data):
    d = data.split('_')
    return d[0],d[1],int(d[2])

def getSpeakerSentenceWordId(ids):
    speakers=dict()
    for i in range(ids.size):
        id_=ids[i]
        sp,se,wo=SpSeWo(id_)
        if not speakers.has_key(sp):
            speakers[sp]=dict()
        if not speakers[sp].has_key(se):
            speakers[sp][se]=dict()
        speakers[sp][se][wo]=i
    return speakers
    
# Input should be numbers, not strings
def getPhonemeIds(ph48c):
    phonemes=dict()
    for i in range(len(ph48c)):
        ph=ph48c[i]
        if not phonemes.has_key(ph):
            phonemes[ph]=list()
        phonemes[ph].append(i)
    return phonemes
    
