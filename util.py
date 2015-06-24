# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:53:44 2015

@author: jan
"""
import sys
def startprogress(title):
    """Creates a progress bar 40 chars long on the console
    and moves cursor back to beginning with BS character"""
    sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
    sys.stdout.flush()


def progress(x):
    """Sets progress bar to a certain percentage x.
    Progress is given as whole percentage, i.e. 50% done
    is given by x = 50"""
    x = int(x * 40 // 100)                      
    sys.stdout.write("#" * x + "-" * (40 - x) + "]" + chr(8) * 41)
    sys.stdout.flush()


def endprogress():
    """End of progress bar;
    Write full bar, then move to next line"""
    sys.stdout.write("#" * 40 + "]\n")
    sys.stdout.flush()
    
import numpy as np
def loadNpy(f):
    with open(f,'r') as fi:
        return np.load(fi)
    
    