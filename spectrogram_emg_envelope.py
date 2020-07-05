#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert
from scipy import signal
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import pandas as pd
 


# In[3]:


q1 = numpy.loadtxt('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/S1/S1.csv')


# In[4]:


q = q1[9,1462700:1502700]
q_abs = numpy.absolute(q)


# In[5]:


def emg_filter_bandpass(x, order=4, sRate=1000., cut=.5, btype='low'):
    """ Forward-backward (high- or low-)pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = cut / nyq
    b, a = butter(order, low, btype=btype, analog=False)
    return filtfilt(b=b, a=a, x=x, axis=0, method='pad', padtype='odd',
                    padlen=numpy.minimum(3 * numpy.maximum(len(a), len(b)), x.shape[0] - 1))


# In[6]:


def calc_hilbert(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = numpy.abs(analytic_signal)
    instantaneous_phase = numpy.unwrap(numpy.angle(analytic_signal))
    instantaneous_frequency = (numpy.diff(instantaneous_phase) / (2.0 * numpy.pi) * 650)
    return [amplitude_envelope, instantaneous_frequency]


# In[7]:


def calc_envelope(signal, freq=20, smooth=21):
    h = calc_hilbert(emg_filter_bandpass(signal, cut=freq))  # lowpass filter & hilbert transform
    sf = savgol_filter(h[0], smooth, 1, axis=0) # extra smoothing
    
    return sf 


# In[8]:


window_size = 500
i = 0
moving_averages = []

while i < len(q_abs) - window_size + 1:
    this_window = q_abs[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 50


# In[21]:


moving_averages=numpy.array(moving_averages)


# In[72]:


f,t, Sxx = signal.spectrogram(moving_averages, fs=280, return_onesided=True, mode='phase')


# In[73]:


plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.savefig("C:/Users/hp/Downloads/spectogram.jpg")
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(13,4))

ax1.plot(q_abs, 'r', label='raw data')
ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
ax2.plot(moving_averages, 'b', label='envelope')

fig.tight_layout()  
plt.legend(loc="upper left")

#plt.savefig("/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/RAW_w200s50.jpg", Transparent = True)
plt.show()

