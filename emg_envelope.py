import matplotlib.pyplot as plt
import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert

q1 = numpy.loadtxt('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/S2.csv', delimiter = ',', unpack = True)

q = q1[9,:5000]
q_abs = numpy.absolute(q)

#envelope functions    
def emg_filter_bandpass(x, order=4, sRate=1000., cut=5., btype='low'):
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

def calc_envelope(signal, freq=20, smooth=21):
    h = calc_hilbert(emg_filter_bandpass(signal, cut=freq))  # lowpass filter & hilbert transform
    sf = savgol_filter(h[0], smooth, 1, axis=0) # extra smoothing
    return sf  


fig, ax = plt.subplots(figsize=(12,4))

ax.plot(q[1500:2500], 'r', label='raw data')
ax.plot(calc_envelope(q_abs[1500:2500]), 'b', label='envelope')

#plt.savefig("/Users/vyshakv/Downloads/EMGdata_images/EMGenv32.jpg", Transparent = True)
plt.show()

