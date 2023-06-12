# Table of contents
* [Introduction](#introduction)
* [Vibrator Sweep](#vibrator-sweep)
* [Installing Xming Server](#installing-xming-server)
* [Installing Seismic Unix](#installing-seismic-unix)
* [Seismic processing of 2D line](#seismic-processing-of-2D-line)
*

## Introduction
The seismic wavelet is the combination of the wavelet transmitted into the earth, modified by the earth’s transmission, and then by the instruments responses. Then, the wavelet processing describes what is done to alter the wavelet so that it is short, well-behaved, and more useful for interpretation.
I will introduce some basics on seismic wavelet processing showing Python codes examples using the seismic sweep from a Land vibrator.

## Vibrator Sweep
An encoded swept-frequency signal (called pilot sweep) transmitted from the vibrator control unit (VCU) in the recording truck to similar units in each vibrator truck.
For the following steps I used a time series sweep in '.csv' format can be found in ./data folder.

The sweep parameters are the following:
- sweep length: 14 seconds
- start frequency: 6 Hz
- end frequency: 95 Hz

Function below simply plot the sweep using Matplotlib Python library.
```Python
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_sweep(file):
    plt.style.use('seaborn-poster')
    %matplotlib inline
    df = pd.read_csv(file)
    plt.figure(figsize = (16, 4))
    plt.plot(df['TIME'], df['AMPLITUDE'])
    plt.xlabel('time(ms)')
    plt.ylabel('amplitude')
    plt.show()
plot_sweep('sweep.csv')
```
<img src=

