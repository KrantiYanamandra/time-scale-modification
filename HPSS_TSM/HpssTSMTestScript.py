#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:48:31 2018

@author: kranthiyanamandra
"""

import librosa
import sounddevice as sd
import HpssTSM

y, sr = librosa.load('../../music/carnival.mp3', sr = None, mono = False, duration = 80)
factor = 1.1

y_scaled = HpssTSM.tsm(y, factor)
#librosa.output.write_wav('results/bassnoteHP.wav',y_scaled,sr)
sd.play(y_scaled,sr)
