#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:52:11 2018

@author: kranthiyanamandra
"""

# standard imports
import numpy as np
import scipy

# Method for time scale modification
def tsm(y, factor):
    
    # check if mono or stereo
    if (len(y.shape) < 2):
       y = np.expand_dims(y, axis=0)
    if(y.shape[0]<3):
        y = y.T
    num_of_channels = y.shape[1]
    
    syn_hop = 512
    L = 2048
    w = np.sin(np.pi*(np.arange(0,L)/L))
    win_len_half = np.round(len(w)//2)
    
    # Anchor points is a 2x2 matrix containing the starting and ending positions of the original and scaled audio    
    anchor_points = np.array([[1, 1], [len(y), np.ceil(factor*len(y))]])
        
    print("Starting stretch")
    
    # Calculations for analysis and synthesis window positions
    output_length = int(np.ceil(factor*len(y)))
    syn_win_pos = np.arange(1,output_length+win_len_half,syn_hop)
    interp_func = scipy.interpolate.interp1d(anchor_points[:,1], anchor_points[:,0],fill_value="extrapolate")
    ana_win_pos = np.round(interp_func(syn_win_pos))
    ana_hop = np.array([ ana_win_pos[1::] - ana_win_pos[0:-1]])  
    ana_hop = np.insert(ana_hop,0,0)
    y_out = np.zeros((output_length,num_of_channels))
    num_of_frames = -1
    
    # Loop through each channel individually
    for channel in range(num_of_channels):
        xC = y[:,channel]
        
        # Apply STFT on selected channel
        X = stft(xC, ana_win_pos, w, num_of_frames)
        
        Y = np.zeros((len(X),len(X.T)),dtype=complex)
        Y[:,0]=X[:,0]
        N = len(w)
        k = np.arange(0,N/2 + 1)
        omega = 2*np.pi*k/N
        
        # Phase vocoder
        for i in range(1, len(X.T)):
                
            dphi = omega * ana_hop[i]
            phCurr = np.angle(X[:,i])
            phLast = np.angle(X[:,i-1])
            hpi = (phCurr - phLast) - dphi
            hpi = hpi - 2 * np.pi * np.round(hpi/(2*np.pi))
            ipa_sample = (omega+hpi/ana_hop[i])
            ipa_hop = ipa_sample * syn_hop
            phSyn = np.angle(Y[:,i-1])
            
            # If no peaks are found, apply standard phase rotation
            
            if(findPeaks(X[:,i])==0):
                
                theta = phSyn+ipa_hop-phCurr
                
                phasor = np.exp(1j*theta)
                Y[:,i] = phasor*X[:,i]
            
            # If peaks are found, rotate only peak samples' phases
            else:
                
                p, irS, irE = findPeaks(X[:,i])
                theta = np.zeros(Y[:,i].size)
                
                for n in range(len(p)):
                    a=int(irS[n])
                    b=int(irE[n])
                    c=int(p[n])
                    
                    theta[a:b] = phSyn[c] + ipa_hop[c] - phCurr[c]
                
                phasor = np.exp(1j*theta)
                    
                Y[:,i] = phasor*X[:,i]
        
        # Apply inverse STFT
        yC = istft(Y, syn_hop, w, output_length)
        y_out[:,channel] = yC
        
    
    #y_out = np.float16(y_out)
    
    return y_out


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
    
    for i in range(num_of_frames2):
        
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
    win_len_half = int(win_len/2)
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
    
    
    ow[ow<10**-3]=1
                
    x = x/ow
        
    x = x[win_len_half:-win_len_half]            
    return x
    

# Method to find peaks in a frame
def findPeaks(stft_spectrum):
    
    magSpec = np.abs(stft_spectrum)
    magSpecPadded = np.pad(magSpec, (2,2), 'constant', constant_values=(0, 0))
    
    peak_loc=np.where(((magSpecPadded[4::] < magSpecPadded[2:-2]) & (magSpecPadded[3:-1] < magSpecPadded[2:-2]) & (magSpecPadded[1:-3] < magSpecPadded[2:-2]) & (magSpecPadded[0:-4] < magSpecPadded[2:-2])))[0]
    
    inflRegionStart = np.zeros(len(peak_loc))
    inflRegionEnd = np.zeros(len(peak_loc))
    
    if (np.all(peak_loc==0)):
        return 0
    
    inflRegionStart[0] = 1
    inflRegionStart[1::] = np.ceil((peak_loc[1::] + peak_loc[0:-1])/2)
    inflRegionEnd[0:-1] = inflRegionStart[1::] - 1
    inflRegionEnd[-1] = len(inflRegionEnd)
    
    return peak_loc, inflRegionStart, inflRegionEnd