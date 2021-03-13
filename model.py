import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from processor import preprocessing

def prep_data(path):
    e = preprocessing(0, 100000)
    d1, d2, d3, d4 = e.loader(path)
    d1 = e.calc_envelope(d1)
    d2 = e.calc_envelope(d2)
    d3 = e.calc_envelope(d3)
    d4 = e.calc_envelope(d4)
    Label = np.array(pd.read_csv(f"{path}S1_labels.csv", engine = 'python')[0:100000])
    #L = Labels
    #Labels = np.reshape(Label, (40000,)
    #e.plot(e.calc_envelope(d2))
    Labels = Label.flatten()
    
    return d1, d2, d3, d4, Labels
    
    
class KNN:
    
    def __init__(self, path):
        d1, d2, d3, d4, Labels = prep_data(path)
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.Labels = Labels
        
            
    def train(self):
        
        #d = {'s1':[np.array(d1)], 's2':[np.array(d2)], 'class':[np.array(Labels[1462700:1502700])]}
        d = np.array([self.d1, self.d2, self.d3, self.d4, self.Labels])
        df = pd.DataFrame(d.T, columns=['s1', 's2', 's3', 'd4', 'class'])
        
        
        X = np.array(df.drop(['class'], 1))
        y = np.array(df['class'])
        
         #'class':[np.array(Labels)]
         
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)
    
        accuracy = clf.score(X_test, y_test)
        print('accuracy =', accuracy)
        
        
#test_signal = np.array([[0.3308264649854933, 0.2381490536075344, 0.1997658380042437, 0.24130765822373523]])
#prediction = clf.predict(test_signal)
    
#print(prediction)

#ind = 1165997
#print(d1[ind], d2[ind], d3[ind], d4[ind])

class CNN:
    
    def __init__(self):