#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:49:58 2018

@author: kranthiyanamandra
"""

import numpy as np
import scipy
import operator

def wsola_tsm(x, factor):
    
    if (len(x.shape) < 2):
       x = np.expand_dims(x, axis=0)
    #x = x.T
    num_of_channels = x.shape[1]
    
    syn_hop = 128
    L = 256
    w = np.sin(np.pi*(np.arange(0,L)/L))**2
    win_len = len(w)
    win_len_half = int(np.round(win_len/2))
    tolerance = 0
    
    if np.isscalar(factor):
        
        anchor_points = np.array([[1, 1], [len(x), np.ceil(factor*len(x))]])
        
    else:
        
        anchor_points = factor
    
    output_length = int(np.ceil(factor*len(x)))
    syn_win_pos = np.arange(1,output_length+win_len_half,syn_hop)
    interp_func = scipy.interpolate.interp1d(anchor_points[:,1], anchor_points[:,0],fill_value="extrapolate")
    ana_win_pos = np.round(interp_func(syn_win_pos))
    ana_win_pos = ana_win_pos.astype(int)
    ana_hop = np.array([ ana_win_pos[1::] - ana_win_pos[0:-1]])  
    ana_hop = np.insert(ana_hop,0,0)
    y_out = np.zeros((output_length,num_of_channels))
    
    min_fac = min(syn_hop / ana_hop)
    
    a = np.zeros((win_len_half+tolerance,num_of_channels))
    b = np.zeros((int(np.ceil(1/min_fac))*win_len+tolerance,num_of_channels))
    x = np.concatenate((a,x,b),axis=0)
    
    ana_win_pos = ana_win_pos + tolerance
    
    for channel in range(num_of_channels):
        
        xC = x[:,channel]
        
        yC = np.zeros(output_length + 2*win_len)
        ow = np.zeros(output_length + 2*win_len)
        
        delta = 0
        
        for i in range(len(ana_win_pos) - 1):
            
            curr_syn_win_ran = np.arange(syn_win_pos[i], syn_win_pos[i] + win_len)
            curr_ana_win_ran = np.arange(ana_win_pos[i] + delta, ana_win_pos[i] + win_len  + delta)
            
            curr_ana_win_ran=curr_ana_win_ran.astype(int)
            
            yC[curr_syn_win_ran] = yC[curr_syn_win_ran] + xC[curr_ana_win_ran] * w
            
            ow[curr_syn_win_ran] = ow[curr_syn_win_ran] + w
            
            
            nat_prog = xC[curr_ana_win_ran + syn_hop]
            
            next_ana_win_ran = np.arange(ana_win_pos[i] - tolerance, ana_win_pos[i] + win_len + tolerance)
            next_ana_win_ran = next_ana_win_ran.astype(int)
            
            x_next_ana_win_ran = xC[next_ana_win_ran]
            
            
            cc = cross_corr(x_next_ana_win_ran, nat_prog, win_len)
            
            max_index, max_value = max(enumerate(cc), key=operator.itemgetter(1))
            
            #max_index = np.argmax(cc)
            
            delta = tolerance - max_index + 2
            
        yC[syn_win_pos[-1]:syn_win_pos[-1]+win_len] = yC[syn_win_pos[-1]:syn_win_pos[-1]+win_len] + xC[ana_win_pos[i-1]+delta:ana_win_pos[i-1]+win_len+delta] * w
        
        ow[syn_win_pos[-1]:syn_win_pos[-1]+win_len] = ow[syn_win_pos[-1]:syn_win_pos[-1]+win_len] + w
        
        for k in range(len(ow)):
            
            if(ow[k] < 10**-3):
                
                ow[k] = 1
        
        yC = yC/ow
        
        yC = yC[win_len_half::]
        yC = yC[0:output_length]
        
        y_out[:, channel] = yC
        
    return y_out
        

def cross_corr(x, y, win_len):
    
    rev_x = x[::-1]
    cc = np.convolve(rev_x,y)
    
    cc = cc[win_len:-win_len+2]
    
    return cc
    
    
    
    
            
            
            
            
            
            
        
    