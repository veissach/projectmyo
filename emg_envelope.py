import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert

q2=pd.read_csv('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/S1/S1.csv')

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
scaled=scaler.fit_transform(q2)
q1=pd.DataFrame(scaled)

q = q1[9,1462700:1502700]
q_abs = numpy.absolute(q)

#envelope functions    
def emg_filter_bandpass(x, order=7, sRate=2000., cut=5., btype='low'):
    """ Forward-backward (high- or low-)pass filtering (IIR butterworth filter) """
    nyq = 0.5 * sRate
    low = cut / nyq
    b, a = butter(order, low, btype=btype, analog=False)
    return filtfilt(b=b, a=a, x=x, axis=0, method='pad', padtype='odd',
                    padlen=numpy.minimum(3 * numpy.maximum(len(a), len(b)), x.shape[0] - 1))

def calc_hilbert(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = numpy.abs(analytic_signal)
    instantaneous_phase = numpy.unwrap(numpy.angle(analytic_signal))
    instantaneous_frequency = (numpy.diff(instantaneous_phase) / (2.0 * numpy.pi) * 650)
    return [amplitude_envelope, instantaneous_frequency]

def calc_envelope(signal, freq=50, smooth=51):
    h = calc_hilbert(emg_filter_bandpass(signal, cut=freq))  # lowpass filter & hilbert transform
    sf = savgol_filter(h[0], smooth, 1, axis=0) # extra smoothing
    return sf  

window_size = 200
i = 0
moving_averages = []

while i < len(q_abs) - window_size + 1:
    this_window = q_abs[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 50
    
'''FOR SEPERATE GRAPHS, USE THIS

fig, ax = plt.subplots(figsize=(13,4))

ax.plot(q_abs, 'r', label='raw data')
ax.plot(moving_averages, 'b', label='envelope')
'''

fig, ax1 = plt.subplots(figsize=(13,4))

ax1.plot(q_abs, 'r', label='raw data')
ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
ax2.plot(moving_averages, 'b', label='envelope')

fig.tight_layout()  
plt.legend(loc="upper left")

#plt.savefig("/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/RAW_w200s50.jpg", Transparent = True)
plt.show()

