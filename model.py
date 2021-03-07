#%%
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from processor import EmgData

e = EmgData(0, 100000)
d1, d2, d3, d4 = e.loader('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/')
d1 = e.calc_envelope(d1)
d2 = e.calc_envelope(d2)
d3 = e.calc_envelope(d3)
d4 = e.calc_envelope(d4)
Label = np.array(pd.read_csv('/Volumes/Seagate Backup Plus Drive/NinaPro DB-2/EMG data/S1_labels.csv')[0:100000])
#L = Labels
#Labels = np.reshape(Label, (40000,)
#e.plot(e.calc_envelope(d2))
Labels = Label.flatten()
#%%

#d = {'s1':[np.array(d1)], 's2':[np.array(d2)], 'class':[np.array(Labels[1462700:1502700])]}
d = np.array([d1, d2, d3, d4, Labels])
df = pd.DataFrame(d.T, columns=['s1', 's2', 's3', 'd4', 'class'])

#%%

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

 #'class':[np.array(Labels)]
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#%%
accuracy = clf.score(X_test, y_test)
print(accuracy)
#%%
