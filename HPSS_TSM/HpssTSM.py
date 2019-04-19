#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:13:53 2018

@author: kranthiyanamandra
"""


# standard imports
import numpy as np
import scipy
import TimeScaleModification as TSM
import WsolaTSM as WTSM

# Method for time scale modification
def tsm(x, factor):
    
    if (len(x.shape) < 2):
       x = np.expand_dims(x, axis=0)
    x = x.T
    num_of_channels = x.shape[1]
    
    
    hps_ana_hop = 256
    L1 = 1024
    hps_win = np.sin(np.pi*(np.arange(0,L1)/L1))**2
    hps_fil_len_harm = 10
    hps_fil_len_perc = 10
    masking_mode = 'binary'

    num_of_frames = -1
    
    x_harm=np.zeros((len(x),num_of_channels))
    x_perc=np.zeros((len(x),num_of_channels))
    
    # Loop through each channel and perform time scale modification
    for channel in range(num_of_channels):
        
        xC = x[:,channel]
        
        # Apply STFT on selected channel
        spec = stft(xC, hps_ana_hop, hps_win, num_of_frames)
        mag_spec = np.abs(spec)
        
        
        magSpecHarm = median_filter(mag_spec,hps_fil_len_harm,2)
        magSpecPerc = median_filter(mag_spec,hps_fil_len_perc,1)
        
        if (masking_mode == 'binary'):
            
            mask_harm = magSpecHarm >  magSpecPerc
            mask_harm = mask_harm.astype(int)
            
            mask_perc = magSpecHarm <= magSpecPerc
            mask_perc = mask_perc.astype(int)
            
        elif (masking_mode == 'relative'):
            
            mask_harm = magSpecHarm / (magSpecHarm + magSpecPerc)
            mask_perc = magSpecPerc / (magSpecHarm + magSpecPerc)
        
        specHarm = mask_harm * spec;
        specPerc = mask_perc * spec;
        
        xHarmC = istft(specHarm, hps_ana_hop, hps_win, int(len(x)))
        xPercC = istft(specPerc, hps_ana_hop, hps_win, int(len(x)))
        
        x_harm[:,channel] = xHarmC
        x_perc[:,channel] = xPercC
        
    y_harm = TSM.tsm(x_harm, factor)
    y_perc = WTSM.wsola_tsm(x_perc, factor)
    
    y = (y_harm+y_perc) * 0.5
    
    return y
    
    

def median_filter(X, length, dim):
    
    Y = np.zeros((len(X),len(X.T)))
    
    if(dim == 1):
        
        
        zeros1 = np.zeros((int(np.floor(length/2)),len(X.T)))
        zeros2 = np.zeros((int(np.ceil(length/2)),len(X.T)))
        
        X_padded = np.concatenate((zeros1,X,zeros2),axis=0)
        
        for i in range(len(X)):
            
            Y[i,:] = np.median(X_padded[i:i+length, :], axis = 0)
            
    if(dim == 2):
        
        zeros3 = np.zeros((len(X),int(np.floor(length/2))))
        zeros4 = np.zeros((len(X),int(np.ceil(length/2))))
        
        X_padded = np.concatenate((zeros3,X,zeros4),axis=1)
        
        for i in range(len(X.T)):
            
            Y[:,i] = np.median(X_padded[:,i:i+length], axis = 1)
            
    return Y
        

# Method to calculate STFT of audio
def stft(x, ana_win_pos, w, num_of_frames1):
    
    win_len = int(len(w))
    win_len_half = int(np.round(win_len/2))
    max_ana_hop = int(np.max(ana_win_pos))
    x_padded = np.pad(x, (win_len_half, win_len+max_ana_hop), 'constant', constant_values=(0, 0))
    
    if np.isscalar(ana_win_pos):
        
        if (num_of_frames1 >= 0):
            num_of_frames2 = num_of_frames1
            
        else:
            num_of_frames2 = int(np.floor((len(x_padded) - win_len)/ana_win_pos + 1))
        
        win_pos = np.arange(0,num_of_frames2 - 1) * ana_win_pos + 1
        
    else:
        
        if (num_of_frames1 >= 0):
            num_of_frames2 = num_of_frames1
        
        else:
            num_of_frames2 = len(ana_win_pos)
        
        win_pos = ana_win_pos[0:num_of_frames2]
    
    spec = np.zeros((win_len_half+1,num_of_frames2), dtype=complex)
    
    win_pos = win_pos.astype(int)
    
    for i in range(num_of_frames2 - 1):
        
        xi = x_padded[win_pos[i]:win_pos[i] + win_len ] * w
        
        Xi = scipy.fft(xi)
        spec[:,i] = Xi[0:win_len_half+1]
        
    
    return spec


# Method to calculate inverse STFT
def istft(spec, syn_hop, w, output_length):
    
    num_of_iter = 1
    num_of_frames = spec[0,:].size
    
    # First iteration
    Yi = spec
    yi = m_istft(Yi, syn_hop, w)
    ana_hopp = syn_hop
    
    # Remaining iterations
    for j in range (1,num_of_iter):
        
        Yi = np.abs(spec)*np.exp(1j * np.angle(stft(yi, ana_hopp, w, num_of_frames)))
        yi = m_istft(Yi, syn_hop, w)
    
    y = yi
    
    y=y[0:output_length]
    
    return y


# Computing the inverse STFT according to the paper 
# "Signal Estimation from Modified Short-Time Fourier Transform" by Griffin and Lim
def m_istft(X, syn_hop, w):
    
    win_len = len(w)
    win_len_half = int(np.round(win_len/2))
    num_of_frames = X[0,:].size
    win_pos = np.arange(0,num_of_frames)*syn_hop 
    signal_length = win_pos[-1] + win_len 
    x = np.zeros(signal_length)
    ow = np.zeros(signal_length)    
    
    for i in range (num_of_frames):
        curr_spec = X[:,i]
        
        Xi_1 = curr_spec
        
        # Add the conjugate complex symmetric upper half of the spectrum
        reverse_curr_spec = curr_spec[::-1]
        reverse_curr_spec = reverse_curr_spec[1:-1]
        conjugate = np.conj(reverse_curr_spec)
        
        Xi = np.concatenate((Xi_1,conjugate),axis=0)
        xi = scipy.ifft(Xi).real
        
        xiw = xi * w
        
        x[win_pos[i]:win_pos[i]+win_len] = x[win_pos[i]:win_pos[i]+win_len] + xiw
        
        ow[win_pos[i]:win_pos[i]+win_len] = ow[win_pos[i]:win_pos[i]+win_len] + w**2
    
    for k in range(len(ow)):
            
        if(ow[k] < 10**-3):
                
            ow[k] = 1
                
    x = x/ow
        
    x = x[win_len_half:-win_len_half]   
                
    return x
    