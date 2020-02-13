#!/bin/python3
import numpy as np
import pickle
from lnm.kneser_ney import kneser_ney
from lnm.witten_bell import witten_bell
from sklearn.model_selection import train_test_split

corpus = pickle.load(open('resources/sentences.pkl','rb'))
X, y = train_test_split(corpus, test_size=0.2)
X_cor = ''
kn_df = []
wb_df = []

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
for line in X:
    X_cor+=line+' '

kn_model = kneser_ney(X_cor)
wb_model = witten_bell(X_cor)

for i in range(1, 2):
    kn_tmp = []
    wb_tmp = []
    for sent in y:
       kn_tmp.append(kn_model.estimate(sent, i))  
       wb_tmp.append(kn_model.estimate(sent, i)) 
    kn_df.append([kn_tmp])
    wb_df.append([wb_tmp])

import matplotlib.pyplot as plt
plt.plot([i for i in range(758)], np.reshape(kn_df[0], (758,)), 'r')
plt.plot([i for i in range(758)], np.reshape(wb_df[0], (758,)), 'b')
plt.show()
