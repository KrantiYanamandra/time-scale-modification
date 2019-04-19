#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:52:11 2018

@author: kranthiyanamandra
"""

import librosa
import TimeScaleModification as TSM

y,sr = librosa.load('../../music/perc1.wav', sr=None, mono=False, duration=30)
factor = 1.1
y_out = TSM.tsm(y, factor)
y_out = y_out
print("Done!")
