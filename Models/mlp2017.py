"""
Implementation of MLP algorithm using CICDos2019 data set
Created on October 29 2020
By Maritza Rosales H.
All rights reserved
"""

from numpy import array, save, asarray
from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dense, Dropout, GRU
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import scikitplot as skplt
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from numpy.random import seed
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score,roc_auc_score
seed(7)

def split_sequence(sequence, n_steps, classes):
    X, y = [], []
    for i in range(0, len(sequence), n_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = array(classes[i:end_ix])
        if np.sum(seq_y)==0:
            seq_y = 0; # normal
        else:
            seq_y = seq_y[seq_y>0]                 # ignore normal events 
            seq_y = np.argmax (np.bincount(seq_y)) # select most frequent attack
        X.append(seq_x)
        y.append(seq_y) 
    return array(X), array(y)

from google.colab import drive 
drive.mount('/content/gdrive')

F = pd.read_csv('gdrive/My Drive/Datasets/2017/dataTrain.csv')
#datasetTest = pd.read_csv('gdrive/My Drive/Datasets/2019/dataTest_with_NormalTraffic.csv')

F.columns

xtrainlabel = F.loc[:,['Label']].values

xtrain = F.loc[:, ['Flow.Duration', 'Tot.Fwd.Pkts', 'Tot.Bwd.Pkts', 'TotLen.Fwd.Pkts',
       'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max', 'Fwd.Pkt.Len.Min',
       'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max', 'Bwd.Pkt.Len.Min',
       'Bwd.Pkt.Len.Std', 'Flow.Byts.s', 'Flow.Pkts.s', 'Flow.IAT.Mean',
       'Flow.IAT.Std', 'Flow.IAT.Max', 'Flow.IAT.Min', 'Fwd.IAT.Mean',
       'Fwd.IAT.Std', 'Fwd.IAT.Min', 'Bwd.IAT.Tot', 'Bwd.IAT.Mean',
       'Bwd.IAT.Std', 'Bwd.IAT.Max', 'Bwd.IAT.Min', 'Fwd.PSH.Flags',
       'Fwd.Pkts.s', 'Bwd.Pkts.s', 'Pkt.Len.Min', 'Pkt.Len.Max',
       'Pkt.Len.Mean', 'Pkt.Len.Std', 'Pkt.Len.Var', 'FIN.Flag.Cnt',
       'SYN.Flag.Cnt', 'RST.Flag.Cnt', 'PSH.Flag.Cnt', 'Down.Up.Ratio',
       'Bwd.Pkts.b.Avg', 'Bwd.Blk.Rate.Avg', 'Subflow.Fwd.Pkts',
       'Subflow.Fwd.Byts', 'Subflow.Bwd.Byts', 'Init.Fwd.Win.Byts',
       'Init.Bwd.Win.Byts', 'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std',
       'Idle.Min']].values

print(xtrain.shape)

# Feature scaling> standarize data
fit = StandardScaler().fit(xtrain)
xtrain = fit.transform(xtrain)
joblib.dump(fit, "gdrive/My Drive/Datasets/scaler_cic2017.joblib")

#PCA for 85% 
pca = PCA(.85)
pca.fit(xtrain)
pca.n_components_ 
joblib.dump(pca, "gdrive/My Drive/Datasets/pca.joblib") 
print(pca.n_components_)

# set number of features
n_features = xtrain.shape[1]
# set number of time steps
n_steps = 1
# split into [samples, timesteps, features]
x_train, y_train = split_sequence(xtrain, n_steps, xtrainlabel)
print(n_features)

print(x_train.shape)
print(y_train.shape)

# define parameters
batch_size = 300 # samples per stack
epochs = 80
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

model = Sequential()
model.add(Dense(15, input_shape=(n_steps, n_features), activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9,activation='softmax'))

#optimizer
adam = Adam(lr=1, beta_1=0.9,beta_2=0.999, epsilon=1e-08)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size, epochs=epochs)

# demonstrate verify fitting on training
yhat = model.predict(x_train)
yyhat = []
for k in range (0,len(yhat)):
    yh = np.argmax(yhat[k]) # max probability
    yyhat.append(yh)
plt.figure(1)
plt.plot(yyhat,'r', y_train,'b')

# TESTING MODEL WITH NEW DATA
# load testing data
F_test  = pd.read_csv('gdrive/My Drive/Datasets/2017/dataTest.csv')

y_test = F_test.loc[:,['Label']].values

x_test = F_test.loc[:, ['Flow.Duration', 'Tot.Fwd.Pkts', 'Tot.Bwd.Pkts', 'TotLen.Fwd.Pkts',
       'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max', 'Fwd.Pkt.Len.Min',
       'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max', 'Bwd.Pkt.Len.Min',
       'Bwd.Pkt.Len.Std', 'Flow.Byts.s', 'Flow.Pkts.s', 'Flow.IAT.Mean',
       'Flow.IAT.Std', 'Flow.IAT.Max', 'Flow.IAT.Min', 'Fwd.IAT.Mean',
       'Fwd.IAT.Std', 'Fwd.IAT.Min', 'Bwd.IAT.Tot', 'Bwd.IAT.Mean',
       'Bwd.IAT.Std', 'Bwd.IAT.Max', 'Bwd.IAT.Min', 'Fwd.PSH.Flags',
       'Fwd.Pkts.s', 'Bwd.Pkts.s', 'Pkt.Len.Min', 'Pkt.Len.Max',
       'Pkt.Len.Mean', 'Pkt.Len.Std', 'Pkt.Len.Var', 'FIN.Flag.Cnt',
       'SYN.Flag.Cnt', 'RST.Flag.Cnt', 'PSH.Flag.Cnt', 'Down.Up.Ratio',
       'Bwd.Pkts.b.Avg', 'Bwd.Blk.Rate.Avg', 'Subflow.Fwd.Pkts',
       'Subflow.Fwd.Byts', 'Subflow.Bwd.Byts', 'Init.Fwd.Win.Byts',
       'Init.Bwd.Win.Byts', 'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std',
       'Idle.Min']].values


x_test = fit.transform(x_test)

# split into [samples, timesteps, features]
x_test, y_test = split_sequence(x_test, n_steps, y_test)

print(x_test.shape)
print(y_test.shape)

# test model on testing data 
yhat = model.predict(x_test)
y_pred = []
for k in range (0,len(yhat)):
    yh = np.argmax(yhat[k])
    y_pred.append(yh)
plt.figure(0)
plt.plot(y_pred,'r', y_test,'b')  

accuracy_tb = metrics.accuracy_score(y_test, y_pred)
print(accuracy_tb)

#TWO CLASSES 
TP=TN=FP=FN=0
for k in range (0,len(y_test)):
    ylabel     = y_test[k];
    ypredicted = y_pred[k];
    if ypredicted == 0 and ylabel == 0:
        TN = TN + 1
    elif ypredicted > 0 and ylabel > 0:
        TP = TP + 1
    elif ypredicted > 0 and ylabel == 0:
        FP = FP + 1
    elif ypredicted == 0 and ylabel > 0:
        FN = FN + 1
    else:
        print('any')
print("ACCURACY", (TP+TN)/(TP+TN+FP+FN))
print("F1-SCORE", (2*TP)/(2*TP+FP+FN))
print("FALSE POS. RATE (FPR)", (FP)/(FP+TN))
print("RECALL (TPR)", (TP)/(TP+FN))
print("Precision ", (TP)/(TP+FP))

#MULTIPLE CLASSES
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred, average='weighted'))
print("F1-SCORE", f1_score(y_test, y_pred, average='weighted'))
print("RECALL (TPR)", recall_score(y_test, y_pred, average='weighted'))

skplt.metrics.plot_confusion_matrix(y_test,y_pred,normalize="True")

model.save('gdrive/My Drive/Datasets/2017/MLP2017')

plot_model(model, show_shapes=True, show_layer_names=True)