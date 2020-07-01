import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert
#from zero_offset import z2

q1 = numpy.loadtxt('/Users/vyshakv/Docs/EMG_2Chs/EMG-S1/HC-1.csv', delimiter = ',', unpack = True)
q = q1[1,:]
#zero offset
def channel(q):

   #find length of the dataset
   lengthSF = len(q)

   #average of the values in the dataset
   a_SF = numpy.average(q) 

   #creating array of average values
   mask1 = numpy.full((lengthSF,), (a_SF))
   
   #zero ofsetting 
   neww_SF = numpy.subtract(q, mask1)
   
   return neww_SF

z = channel(q)
z_abs = numpy.absolute(z)

#MAV windows
window_size = 1000
i = 0
moving_averages = []

while i < len(z_abs) - window_size + 1:
    this_window = z_abs[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1
    
w = numpy.array([moving_averages])  
wt = w.T  

#envelope functions    
def emg_filter_bandpass(x, order=4, sRate=650., cut=15., btype='low'):
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

def calc_envelope(signal, freq=20, smooth=51):
    filtered_emg_signal = emg_filter_bandpass(signal, cut=freq)  # lowpass filter
    h = calc_hilbert(filtered_emg_signal)  # hilbert transform
    sf = savgol_filter(h[0], smooth, 1, axis=0)
    return sf  # extra smoothing


#plt.plot(calc_envelope(q))

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
#ax1.plot(calc_envelope(wt[7500:8500]), 'r')
#ax2.plot(wt[7500:8500], 'b')
#ax3.plot(q[7500:8500], 'g')

fig, ax1 = plt.subplots()

ax1.plot(z[:1000], 'r')

ax2 = ax1.twinx()

ax2.plot(calc_envelope(wt[:1000]), 'b')

fig.tight_layout()

#plt.savefig("/Users/vyshakv/Downloads/EMGdata_images/EMGenv32.jpg", Transparent = True)
plt.show()
