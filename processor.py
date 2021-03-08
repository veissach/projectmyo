#import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from scipy.signal import butter, filtfilt, savgol_filter, hilbert
from sklearn.preprocessing import MaxAbsScaler

class preprocessing:#envelope functions  
    
    
    def __init__(self, lRange, uRange, subject_num: int = 1, channels: int = 2):
        self.subject_num = subject_num
        self.channels = channels
        self.lRange = lRange
        self.uRange = uRange

        
    def emg_filter_bandpass(self, x, order=4, sRate=2000., cut=5., btype='low'):
        """ Forward-backward (high- or low-)pass filtering (IIR butterworth filter) """
        nyq = 0.5 * sRate
        low = cut / nyq
        b, a = butter(4, low, btype=btype, analog=False)
        return filtfilt(b=b, a=a, x=x, axis=0, method='pad', padtype='odd',
                        padlen=numpy.minimum(3 * numpy.maximum(len(a), len(b)), x.shape[0] - 1))
    
    def calc_hilbert(self, signal): #preprocessing
        analytic_signal = hilbert(signal)
        amplitude_envelope = numpy.abs(analytic_signal)
        instantaneous_phase = numpy.unwrap(numpy.angle(analytic_signal))
        instantaneous_frequency = (numpy.diff(instantaneous_phase) / (2.0 * numpy.pi) * 650)
        return [amplitude_envelope, instantaneous_frequency]
    
    def calc_envelope(self, signal, freq=5., smooth=51): #preprocessing
        h = self.calc_hilbert(self.emg_filter_bandpass(signal, cut=freq))  # lowpass filter & hilbert transform
        #h = numpy.array(h)
        sf = savgol_filter(h[0], smooth, 1, axis=0) # extra smoothing
        return sf  
    
    def MAV(self, signal): 
        sig = self.calc_envelope(signal)
        window_size = 512
        i = 0
        moving_averages = []
    
        while i < len(sig) - window_size + 1:
            this_window = sig[i : i + window_size]
                    
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += 1
        return moving_averages
    
    def sum_env(self, signal1, signal2):
        env_sum = [signal1[i] + signal2[i] for i in range(len(signal1))] 
        
        return env_sum    
    
    def loader(self, path):
        num = self.subject_num
        data = numpy.loadtxt(f"{path}S{num}.csv", delimiter = ',', unpack = True)
        scaler = MaxAbsScaler()
        scaled = scaler.fit_transform(data)
        _scaled1 = numpy.absolute(scaled[8,self.lRange:self.uRange])
        _scaled2 = numpy.absolute(scaled[9,self.lRange:self.uRange])
        _scaled3 = numpy.absolute(scaled[6,self.lRange:self.uRange])
        _scaled4 = numpy.absolute(scaled[3,self.lRange:self.uRange])

        '''    
        for chan in range(1, channels+1):
            _data = []
            _data.append(numpy.absolute(scaled[8,lRange:uRange]))
            chan += 1
        '''    
        return _scaled1, _scaled2, _scaled3, _scaled4
    
    def data_prep(self, path):
        _data1, _data2 = self.loader(path)
        
        mav1 = numpy.asarray(self.MAV(_data1))
        mav2 = numpy.asarray(self.MAV(_data2))
        
        return self.sum_env(mav1, mav2)
    
    def plot(self, data1):
    #def plot(self, data1, data2):
        
        #'''FOR SEPERATE GRAPHS, USE THIS
        
        fig, ax = plt.subplots(figsize=(20,4))
    
        ax.plot(data1, 'r', label='raw data')
        #ax.plot(data2, 'b', label='envelope')
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
    
    
    #subject_num: int = 1
    #channels: int = 2
    #lRange = 1462700
    #uRange = 1502700
    
    #s1, s2 = loader(uRange, lRange)
    #s1 = s1
    #s2 = s2
    #mav1 = numpy.array(MAV(s1))
    #mav2 = numpy.array(MAV(s2))
    #sum_ = numpy.array(sum_env(mav1, mav2)[2000:6000])
    
    #plot(s1, sum_env(mav1, mav2)[2000:6000])
    
if __name__ == '__main__':
    path = '/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/'
    
    e = EmgData(2000, 3000)
    
    s1, s2 = e.loader(path)
    s1 = s1
    s2 = s2
    
    e.plot(s1, e.data_prep(path))
    
    
            
            
            
            
