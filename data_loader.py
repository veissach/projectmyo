from processor import EmgData

e = EmgData(1462700, 1502700)
d1, d2 = e.loader('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/')
e.plot(e.calc_envelope(d2))