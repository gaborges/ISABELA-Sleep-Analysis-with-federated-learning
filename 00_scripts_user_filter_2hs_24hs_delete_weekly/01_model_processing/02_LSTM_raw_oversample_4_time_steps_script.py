#!/usr/bin/env python
# coding: utf-8

# In[14]:


# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow import keras


import tensorflow_addons as tfa
import tensorflow as tf

from scipy import stats

import datetime;
import warnings
warnings.filterwarnings("ignore")


# In[15]:


# configs
#EPOCHS = 5
BATCH_SIZE = 32

outputMetricFile = "result_trad_LSTM_over1.csv"

TIME_SERIES_SIZE = 4   # Determines the window size. Ex (4,9)
TIME_STEP_SHIFT  = 1   # Determines specifies the number of steps to move the window forward at each iteration.

baseFolder = "../data_2019_processed/"

# selected features
inputFeatures = ["activity","location","day_of_week",
                 "light","phone_lock","proximity",
                 "sound","time_to_next_alarm", "minutes_day"]
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]

NN_type = 'LSTM'
UNITS_NUMBER = 128
EPOCHS_ARRAY_TEST = [2,7,15,30,50,100,120]
EPOCHS_ARRAY_TEST = [150,200,300]


# In[16]:


# y_test     = Array with real values
# yhat_probs = Array with predicted values
def printMetrics(y_test,yhat_probs):
    # predict crisp classes for test set deprecated
    #yhat_classes = model.predict_classes(X_test, verbose=0)
    #yhat_classes = np.argmax(yhat_probs,axis=1)
    yhat_classes = yhat_probs.round()
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    print("\Confusion Matrix")
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    
    array = []
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['cohen_kappa_score'] = kappa
    results['roc_auc_score'] = auc
    results['matrix'] = np.array(matrix,dtype=object)
    results['TP'] = matrix[0][0]
    results['FP'] = matrix[0][1]
    results['FN'] = matrix[1][0]
    results['TN'] = matrix[1][1]
    
    array.append(accuracy)
    array.append(precision)
    array.append(recall)
    array.append(f1)
    array.append(kappa)
    array.append(auc)
    array.append(np.array(matrix,dtype=object))
    array.append(matrix[0][0]) # TP
    array.append(matrix[0][1]) # FP
    array.append(matrix[1][0]) # FN
    array.append(matrix[1][1]) # TN
    
    return results, array

def showGlobalMetrics(metrics):
    accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score = 0,0,0,0,0,0
    for metric in metrics:
        accuracy = accuracy + metric['accuracy']
        precision = precision + metric['precision']
        recall = recall + metric['recall']
        f1_score = f1_score + metric['f1_score']
        cohen_kappa_score = cohen_kappa_score + metric['cohen_kappa_score']
        roc_auc_score = roc_auc_score + metric['roc_auc_score']
        
    # mean
    size = len(metrics)
    print(size)
    accuracy = accuracy / size
    precision = precision / size
    recall = recall / size
    f1_score = f1_score / size
    cohen_kappa_score = cohen_kappa_score / size
    roc_auc_score = roc_auc_score / size
    
    #show:\
    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1_score: ",f1_score)
    print("cohen_kappa_score: ",cohen_kappa_score)
    print("roc_auc_score: ",roc_auc_score)
    
    return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score]
    
def transform_data_type(dataframe):
    
    # transform inputs
    for column in inputFeatures:
        dataframe[column] = dataframe[column].astype('float32')
    
    # transform outputs
    for column in outputClasses:
        dataframe[column] = dataframe[column].astype('float32')
    
    return dataframe

# one-hot encoding function
def transform_output_nominal_class_into_one_hot_encoding(dataset):
    # create two classes based on the single class
    one_hot_encoded_data = pd.get_dummies(dataset['class'])
    #print(one_hot_encoded_data)
    dataset['awake'] = one_hot_encoded_data['awake']
    dataset['asleep'] = one_hot_encoded_data['asleep']
    
    return dataset

# one-hot encoding function
def transform_output_numerical_class_into_one_hot_encoding(dataset):
    # create two classes based on the single class
    one_hot_encoded_data = pd.get_dummies(dataset['class'])
    #print(one_hot_encoded_data)
    dataset['awake'] = one_hot_encoded_data[0]
    dataset['asleep'] = one_hot_encoded_data[1]
    
    return dataset


def create_dataset_time_series_with_one_output(X, y, window_time_steps=1, shift_step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - window_time_steps, shift_step):
        v = X.iloc[i:(i + window_time_steps)].values
        labels = y.iloc[i: i + window_time_steps]
        Xs.append(v)        
        ys.append(stats.mode(labels)[0][0])
        
    if len(y.columns) == 1:
        return np.array(Xs), np.array(ys).reshape(-1, 1)
    else:
        return np.array(Xs), np.array(ys).reshape(-1, len(y.columns))


# In[17]:


#X_train = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized.csv")
X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized.csv")
#X_train = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized_balanced_undersample.csv")
X_train  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized_balanced_oversample.csv")

#AA = pd.read_csv(baseFolder+"allData-classification-numeric-normalized.csv")
#X_train, X_test = train_test_split(AA,test_size=0.25)


# In[18]:


print(X_train.info())
X_train


# In[6]:


# transform output to one_hot_encoding for the testing dataset
X_test = transform_output_nominal_class_into_one_hot_encoding(X_test)

# transform output to one_hot_encoding for the testing dataset
X_train = transform_output_nominal_class_into_one_hot_encoding(X_train)


# transforms the input data to float32
X_test = transform_data_type(X_test)

# transforms the input data to float32
X_train = transform_data_type(X_train)


# In[7]:


# selects the data to train and test
X_train_data = pd.DataFrame(data=X_train,columns=inputFeatures)
y_train_data = pd.DataFrame(data=X_train,columns=outputClasses)
# selec test dataset (fixed to all)
X_test_data = pd.DataFrame(data=X_test,columns=inputFeatures)
y_test_data = pd.DataFrame(data=X_test,columns=outputClasses)

X_train_data, y_train_data = create_dataset_time_series_with_one_output(   #timestamp
    X_train_data, 
    y_train_data, 
    TIME_SERIES_SIZE, 
    TIME_STEP_SHIFT
)

X_test_data, y_test_data = create_dataset_time_series_with_one_output(    #timestamp
    X_test_data, 
    y_test_data, 
    TIME_SERIES_SIZE, 
    TIME_STEP_SHIFT
)


print("shape: ",X_train_data.shape, y_train_data.shape)
print("Size: ",X_test_data.shape,y_test_data.shape)       


# In[8]:


# transtorm data to tensor slices
test_dataset_series = tf.data.Dataset.from_tensor_slices((X_test_data, y_test_data))
train_dataset_series = tf.data.Dataset.from_tensor_slices((X_train_data, y_train_data))


# In[9]:


train_dataset_series


# In[10]:


#client_test_dataset.window(size=4, shift=1, stride=1, drop_remainder=True)
#test_dataset_series.batch(BATCH_SIZE) # usado no federated learning


# In[11]:


# batch data size
#test_dataset_series1 = test_dataset_series.batch(1)
#train_dataset_series1 = train_dataset_series.batch(1)

test_dataset_series1 = test_dataset_series.batch(BATCH_SIZE)
train_dataset_series1 = train_dataset_series.batch(BATCH_SIZE)

train_dataset_series1


# In[12]:


columnsOutputMetrics = ['NN_type','units','epochs','batch_size','window_size','time_step_shift',
           'start_time','end_time','time_s','time_m',
           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
           'TP','FP','FN','TN']

allMetrics = []


# In[21]:


for TEST_EPOCHS in EPOCHS_ARRAY_TEST:
    # general data from the run
    generalData = [NN_type,UNITS_NUMBER,TEST_EPOCHS,BATCH_SIZE,TIME_SERIES_SIZE,TIME_STEP_SHIFT]
    
    print("input_shape=[", X_train_data.shape[1], X_train_data.shape[2],"]")
    print("output shape:",len(outputClasses))
    print("Epochs:",TEST_EPOCHS)
    
    current_time = datetime.datetime.now()
    time_stamp = current_time.timestamp()
    print("Start timestamp:", time_stamp,current_time)
    print()

    verbose, epochs, batch_size = 0, TEST_EPOCHS, BATCH_SIZE
    #verbose, epochs, batch_size = 1, 3, 16
    model = keras.Sequential()
    model.add(LSTM(UNITS_NUMBER,activation="tanh", 
              input_shape=[X_train_data.shape[1], X_train_data.shape[2]]))
    #model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(len(outputClasses), activation='softmax'))#softmax,sigmoid
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
              #loss='binary_crossentropy',loss='categorical_crossentropy',
              #loss='binary_crossentropy',  sparse_categorical_crossentropy     
    # fit network
    hist = model.fit(train_dataset_series1, epochs=epochs, verbose=verbose, batch_size=batch_size) #, batch_size=batch_size, validation_split=0.1
    # evaluate model
    print(hist)
    #accuracy = model.evaluate(test_dataset_series1) # , batch_size=batch_size, verbose=0
    # predict
    yhat_probs = model.predict(X_test_data,verbose=0)
    # predict crisp classes for test set deprecated
    
    # generate time metrics
    current_time2 = datetime.datetime.now()
    time_stamp2 = current_time2.timestamp()
    processing_time_s = (time_stamp2-time_stamp)
    # generate general metrics
    rowData = [current_time,current_time2,processing_time_s,(processing_time_s)/60]

    y_pred_labels = pd.DataFrame(data=yhat_probs,columns=['awake','asleep'])
    y_test_labels = pd.DataFrame(data=y_test_data,columns=['awake','asleep'])

    feature_metrics_gathered = []
    # print('')
    print('awake')    
    res,resA = printMetrics(y_test_labels['awake'],y_pred_labels['awake'])
    feature_metrics_gathered.append(res)
   
    #columns = ['NN_type','units','epochs','batch_size','max_iterations',''Users',
    #            round_iteration','start_time','end_time','round_time_s','round_time_m',
    #           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
    #           'TP','FP','FN','TN']
    # new data
    classData = np.concatenate((['awake'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    
    print('')
    print('asleep')
    res,resA = printMetrics(y_test_labels['asleep'],y_pred_labels['asleep'])
    feature_metrics_gathered.append(res)
    # new data
    classData = np.concatenate((['asleep'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    print('')
    print('Global')
    resA = showGlobalMetrics(feature_metrics_gathered) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    # new data
    classData = np.concatenate((['avg'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    print('')
    print("End timestamp:", time_stamp2,current_time2)
    print('')
    print('')


# In[ ]:


dataMetrics = pd.DataFrame(data=allMetrics,columns=columnsOutputMetrics) 

dataMetrics


# In[ ]:


dataMetrics.to_csv(outputMetricFile, sep=',', encoding='utf-8', index=False)


# In[ ]:




