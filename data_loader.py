from processor import preprocessing
from model import KNN


KNN('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/').train()


#e = preprocessing(1462700, 1502700)
#d1, d2 = e.loader('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/')
#e.plot(e.calc_envelope(d2))