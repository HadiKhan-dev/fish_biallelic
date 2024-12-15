"""
File which contains functions that takes as input a list of
probabalistic haplotypes and use them to simulate a multi generation
progeny of founders made up of these haplotypes
"""
import random
import numpy as np
import pickle
import ujson

def concretify_haps(haps_list):
    """
    Takes a list of probabalistic haps and turns each of them 
    into a list of 0s and 1s by taking the highest probability
    allele at each site
    """
    
    concreted = []
    
    for hap in haps_list:
        concreted.append(np.argmax(hap,axis=1))
    return concreted

def pairup_haps(haps_list,shuffle=False):
    """
    Pair up a list of concrete haps (made up of 0s and 1s)
    """
    
    haps_copy = pickle.loads(pickle.dumps(haps_list))
    
    if shuffle:
        random.shuffle(haps_copy)
    
    num_pairs = len(haps_list)//2
    
    haps_paired = []
    
    for i in range(num_pairs):
        first = haps_copy[2*i]
        second = haps_copy[2*i+1]
        
        haps_paired.append([first,second])
    
    return haps_paired

def recombine_haps(hap_pair,site_locs,recomb_rate=10**-8):
    """
    Takes as input a pair of concrete haps giving the allele
    at each variable site as well as a list of positions for
    the variable sites. The function then creates a composite
    haplotype simulating meiosis by switching over based on
    an exponential distribution
    """
    
    scale = 1/recomb_rate
    
    assert len(hap_pair[0]) == len(hap_pair[1]), "Length of two haplotypes is different"
    assert len(hap_pair[0]) == len(site_locs), "Different length of hap and of list of site locations"
    
    cur_loc = site_locs[0]
#%%
data = sa[1]
    
cm = concretify_haps(data)
pa = pairup_haps(cm)

print(pa)
#%%
print(np.random.exponential(10**8))
