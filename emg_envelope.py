import matplotlib.pyplot as plt
import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert
from sklearn.preprocessing import MaxAbsScaler


#envelope functions    
def emg_filter_bandpass(x, order=4, sRate=2000., cut=5., btype='low'):
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

def calc_envelope(signal, freq=5., smooth=51):
    h = calc_hilbert(emg_filter_bandpass(signal, cut=freq))  # lowpass filter & hilbert transform
    #h = numpy.array(h)
    sf = savgol_filter(h[0], smooth, 1, axis=0) # extra smoothing
    return sf  

def MAV(signal):
    sig = calc_envelope(signal)
    window_size = 500
    i = 0
    moving_averages = []

    while i < len(sig) - window_size + 1:
        this_window = sig[i : i + window_size]
                
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

def sum_env(signal1, signal2):
    env_sum = [signal1[i] + signal2[i] for i in range(len(signal1))] 
    
    return env_sum    

def loader(uRange, lRange):
    num = subject_num
    data = numpy.loadtxt(f"{INPUT_FOLDER}S{num}.csv", delimiter = ',', unpack = True)
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(data)
    _scaled1 = numpy.absolute(scaled[8,lRange:uRange])
    _scaled2 = numpy.absolute(scaled[9,lRange:uRange])

    '''    
    for chan in range(1, channels+1):
        _data = []
        _data.append(numpy.absolute(scaled[8,lRange:uRange]))
        chan += 1
    '''    
    return _scaled1, _scaled2

def data_prep():
    _data1, _data2 = loader(lRange, uRange)
    
    mav1 = MAV(_data1)
    mav2 = MAV(_data2)
    
    return sum_env(mav1, mav2)

def plot(data1, data2):
    
    #'''FOR SEPERATE GRAPHS, USE THIS
    
    fig, ax = plt.subplots(figsize=(10,4))

    #ax.plot(data1, 'r', label='raw data')
    ax.plot(data2, 'b', label='envelope')
    '''
    fig, ax1 = plt.subplots(figsize=(10,4))
    
    ax1.plot(data1, 'r', label='raw data')
    ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
    ax2.plot(data2, 'b', label='envelope')
    
    fig.tight_layout()  
    plt.legend(loc="upper left")
    '''
    #plt.savefig("/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/RAW_w200s50.jpg", Transparent = True)
    plt.show()


INPUT_FOLDER = '/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/'

subject_num: int = 1
channels: int = 2
lRange = 1462700
uRange = 1502700

s1, s2 = loader(uRange, lRange)
s1 = s1
s2 = s2
mav1 = MAV(s1)
mav2 = MAV(s2)


plot(s1, sum_env(mav1, mav2)[2000:6000])

            
        
        
        
