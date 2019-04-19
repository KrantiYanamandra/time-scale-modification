# time-scale-modification

Two different implementations of time scale modification were done. 

* Phase-locked TSM uses identity phase locking. Peak detection is done and phase is preserved to reduce artifacts
* HPSS TSM separates the audio into harmonic and percussive components and uses two algorithms - Phase locked TSM on the harmonic part and WSOLA on the percussive part.
