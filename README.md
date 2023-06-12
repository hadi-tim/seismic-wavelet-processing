# Table of contents
* [Introduction](#introduction)
* [Vibrator Sweep](#vibrator-sweep)
* [FFT in Python](#fft-in-python)
* [Amplitude and Phase spectrum](#amplitude-and-phase-spectrum)
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
<img src=./images/sweep.png>

## FFT in Python
Let's calculate the amplitude and the phase spectrum of our sweep. 
The amplitude and phase spectrums are obtained by converting the signal from time domain to frequency domain.
In programming the **Fourier Transform**  needs to be in a discrete form given by:

$$X(k) = \sum_{n=0}^n x(n) e^{-i2 \pi kn \over N}$$

where, $k$ is the index of the $k_{th}$ frequency point, where $f_k$ = $k f_s \over N$, $f_s$ is the frequency sampling of the signal.
## Amplitude and Phase Spectrum

In the Python code, I justt called the methods of libraries Numpy and Scipy in Python to implement the amplitude and phase spectrums as below:
```Python
from numpy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
%matplotlib inline

dt = 0.002
t = np.arange(0, 14.002, dt)
signal = pd.read_csv('sweep.csv')['AMPLITUDE']
minsignal, maxsignal = signal.min(), signal.max()

## Compute Fourier Transform
n = len(t)
fhat = np.fft.fft(signal, n) #computes the fft
psd = fhat * np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n) #frequency array
idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) #first half index

# Another approch to get the Fourier Transform of s
fd = np.fft.fft(signal) / n * 2
# The amplitude spectrum.
fa = abs(fd)
# The phase spectrum.
fp = np.arctan2(fd.imag, fd.real)
#fp = np.arctan2(fhat.imag, fhat.real)
fs = 1/dt

plt.figure(figsize=(12, 8), dpi=80)
plt.subplot(211)
plt.plot(t, signal, color='r', lw=1, label='Sweep Signal')
plt.grid(linestyle='--', linewidth=0.8, alpha=0.3)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 2, 3)
plt.plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=1, label='Amplitude Spectrum')
plt.grid(linestyle='--', linewidth=0.8, alpha=0.3)
plt.xlim(0, fs/4)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 2, 4)
plt.plot(freq, fp, lw=1, color='g')
plt.grid(linewidth=0.8, alpha=0.3)
plt.xlim(0, 10)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Phase', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()
```
The plots below show the corresponding amplitude and phase spectrums of our sweep.

<img src=./images/amp_phase_spec.png>




