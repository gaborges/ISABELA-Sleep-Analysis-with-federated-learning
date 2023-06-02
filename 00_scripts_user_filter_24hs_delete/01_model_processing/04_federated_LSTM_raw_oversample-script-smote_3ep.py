#!/usr/bin/env python
# coding: utf-8

# In[1]:


outputMetricFile = "result3epochsLSTM_oversample_smote_300ep.csv"

#!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# Bibliotecas Auxiliares
import seaborn as sns
import numpy as np
import collections
import matplotlib.pyplot as plt
import nest_asyncio
nest_asyncio.apply()
import tensorflow_federated as tff
np.random.seed(0)
from collections import Counter

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

import collections
import functools
import os
import time
from scipy import stats

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import datetime;
import warnings
warnings.filterwarnings("ignore")

print(tf.__version__)


# In[2]:


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


# In[3]:


# configs
EPOCHS = 3
NUM_EPOCHS = EPOCHS
BATCH_SIZE = 32

TIME_SERIES_SIZE = 4   # Determines the window size. Ex (4,9)
TIME_STEP_SHIFT  = 1   # Determines specifies the number of steps to move the window forward at each iteration.

baseFolder = "../data_2019_processed/"

# selected features
inputFeatures = ["activity","location","day_of_week",
                 "light","phone_lock","proximity",
                 "sound","time_to_next_alarm", "minutes_day"]
# outputs
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]


# In[4]:


# test data comprising 25% of the data. It must be fixed to all models being evaluated
X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized.csv")

# input folder
inputFolders = baseFolder

            
dsTrain = ['0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
            '0tdmm6rwW3KquQ73ATYYJ5JkpMtvbppJ0VzA2GExdA', 
            '2cyV53lVyUtlMj0BRwilEWtYJwUiviYoL48cZBPBq0', 
            '2J22RukYnEbKTk7t+iUVDBkorcyL5NKN6TrLe89ys', 
            #['5FLZBTVAPwdq9QezHE2sVCJIs7p+r6mCemA2gp9jATk'], #does not have the file
            '7EYF5I04EVqisUJCVNHlqn77UAuOmwL2Dahxd3cA', 
            'a9Qgj8ENWrHvl9QqlXcIPKmyGMKgbfHk9Dbqon1HQP4', 
            'ae4JJBZDycEcY8McJF+3BxyvZ1619y03BNdCxzpZTc', 
            'Ch3u5Oaz96VSrQbf0z31X6jEIbeIekkC0mwPzCdeJ1U', 
            'CH8f0yZkZL13zWuE9ks1CkVJRVrr+jsGdUXHrZ6YeA', 
            'DHO1K4jgiwZJOfQTrxvKE2vn7hkjamigroGD5IaeRc', 
            'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', 
            'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
            'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
            'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
            'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
            'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
            'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
            'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
            'PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc', 
            'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
            'rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw', 
            'RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI', 
            'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4', 
            'VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is', 
            'Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw', 
            'XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA', 
            'YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw', 
            'ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM', 
            'ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']


# client datasets used on the training process
dsTrain =  ['0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
            '0tdmm6rwW3KquQ73ATYYJ5JkpMtvbppJ0VzA2GExdA', 
            '2cyV53lVyUtlMj0BRwilEWtYJwUiviYoL48cZBPBq0', 
            '2J22RukYnEbKTk7t+iUVDBkorcyL5NKN6TrLe89ys', 
            #['5FLZBTVAPwdq9QezHE2sVCJIs7p+r6mCemA2gp9jATk'], #does not have the file
            '7EYF5I04EVqisUJCVNHlqn77UAuOmwL2Dahxd3cA', 
            'a9Qgj8ENWrHvl9QqlXcIPKmyGMKgbfHk9Dbqon1HQP4', 
            'ae4JJBZDycEcY8McJF+3BxyvZ1619y03BNdCxzpZTc', 
            'Ch3u5Oaz96VSrQbf0z31X6jEIbeIekkC0mwPzCdeJ1U', 
            'CH8f0yZkZL13zWuE9ks1CkVJRVrr+jsGdUXHrZ6YeA', 
            'DHO1K4jgiwZJOfQTrxvKE2vn7hkjamigroGD5IaeRc', 
            'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', 
            'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
            'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
            'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
            'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
            'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
            'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
            'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
            #['PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc'], 
            'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
            #['rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw'], 
            #['RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI'], 
            'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4'] 
            #['VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is'], 
            #['Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw'], 
            #['XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA'], 
            #['YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw'], 
            #['ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM'], 
            #['ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']]


# In[ ]:


# load cliend data
clientList = []

# balancing classes
#oversample = RandomOverSampler(sampling_strategy='auto') #minority
oversample = SMOTE(random_state = 32)
#undersample = RandomUnderSampler(sampling_strategy='majority')

for i in range(0,len(dsTrain)):
    print (i," ", str(inputFolders)+"student_"+dsTrain[i]+"_numeric.csv") #_numeric
    # load client data
    dataset = pd.read_csv(inputFolders+"student_"+dsTrain[i]+"_numeric.csv")
    
    # print(dataset)
    y_train = dataset['class'].copy()
        
    # does not add datasets that dont have instances from both classes
    if y_train.sum() != 0 and (y_train.sum() != len(y_train)):
        X_train = dataset
        
        del X_train['class']
        del X_train['timestamp']
        del X_train['timestamp_text']
        # balancing classes
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        #X_train, y_train = undersample.fit_resample(dataset, y_train)
        
        X_train['class'] = y_train
        
        clientList.append(X_train)
        
print("Total",(len(clientList)))


# In[11]:


# undestand the dataset by looking on their infos
print(X_test.info())

X_test


# In[12]:


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

# transform output to one_hot_encoding for the testing dataset
X_test = transform_output_nominal_class_into_one_hot_encoding(X_test)

# transform output to one_hot_encoding for the input dataset
for i in range(0,len(clientList)):
    clientList[i] = transform_output_numerical_class_into_one_hot_encoding(clientList[i])
    #print (clientList[i])


# In[13]:


X_test.info()


# In[14]:


def transform_data_type(dataframe):
    
    # transform inputs
    for column in inputFeatures:
        dataframe[column] = dataframe[column].astype('float32')
    
    # transform outputs
    for column in outputClasses:
        dataframe[column] = dataframe[column].astype('float32')
    
    return dataframe

# transforms the data
X_test = transform_data_type(X_test)

X_test.info()


# In[15]:


# selects the data to train and test
X_test_data = X_test[inputFeatures]
y_test_label = X_test[outputClasses]

# transform to time series
X_test_data, y_test_label = create_dataset_time_series_with_one_output(    #timestamp
    X_test_data, 
    y_test_label, 
    TIME_SERIES_SIZE, 
    TIME_STEP_SHIFT
)

# transtorm data to tensor slices
client_test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data, y_test_label))
#client_test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data.values, y_test_label.values))

#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True)
client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)

print(client_test_dataset.element_spec)
client_test_dataset


# In[16]:


federated_training_data = []
# transform the data
for i in range(0,len(clientList)):
    # selects the data to train and test
    data   = clientList[i][inputFeatures]
    labels = clientList[i][outputClasses]
    # transform data to float32
    #data = transform_data_type(data)

    # transform to time series
    data, labels = create_dataset_time_series_with_one_output(    #timestamp
        data, 
        labels, 
        TIME_SERIES_SIZE, 
        TIME_STEP_SHIFT
    )

    # transform the data to tensor slices
    client_train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    #client_train_dataset = tf.data.Dataset.from_tensor_slices((data.values, labels.values))
    # apply the configs
    client_train_dataset = client_train_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)
    # transform the data to
    federated_training_data.append(client_train_dataset)


# In[17]:


def create_keras_model():
    model = keras.Sequential()
    model.add(LSTM(128,input_shape=(4, 9,))) #return_sequences=True, 
    #model.add(LSTM(128,input_shape=(2, 9,)))
    #model.add(keras.layers.Dropout(rate=0.5))
    #model.add(LSTM(64,activation="tanh"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))#softmax,sigmoid
    
    return model


# In[18]:


keras_model = create_keras_model()
#keras_model.summary()
keras_model.summary()


# In[19]:


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    #keras_model = create_keras_model()
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
      keras_model,
      input_spec=client_train_dataset.element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(), #BinaryCrossentropy
      metrics=[tf.keras.metrics.CategoricalAccuracy()])


# In[20]:


fed_avg_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),#client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


# In[21]:


print(fed_avg_process.initialize.type_signature.formatted_representation())


# In[22]:


state = fed_avg_process.initialize()


# In[22]:


roundData = []

columns = ['NN_type','units','epochs','batch_size','max_iterations','Users',
           'round_iteration','start_time','end_time','round_time_s','round_time_m',
           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
           'TP','FP','FN','TN']

MAX_ITERATIONS = 300
#NUM_EPOCHS
#BATCH_SIZE
NN_type = 'LSTM'
UNITS_NUMBER = 128
USER_NUMBER = len(dsTrain)

generalData = [NN_type,UNITS_NUMBER,NUM_EPOCHS,BATCH_SIZE,MAX_ITERATIONS,USER_NUMBER]

for i in range(0,MAX_ITERATIONS):
    current_time = datetime.datetime.now()
    time_stamp = current_time.timestamp()
    print(i,"Start timestamp:", time_stamp,current_time)
    
    result = fed_avg_process.next(state, federated_training_data[0:19])
    state = result.state
    metrics = result.metrics
    print('round  {}, metrics={}'.format(i,metrics))

    print('')
    #def keras_evaluate(state, round_num):
        # Take our global model weights and push them back into a Keras model to
        # use its standard `.evaluate()` method.
    keras_model = create_keras_model()


    keras_model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # get neural network weights
    weights = fed_avg_process.get_model_weights(state)
    weights.assign_weights_to(keras_model)


    #tttt.model.assign_weights_to(keras_model)
    yhat_probs = keras_model.predict(X_test_data,verbose=0)

    maxX = yhat_probs.max()
    minX = yhat_probs.min()
    avgX = yhat_probs.mean()

    print(maxX,minX,avgX)
    print(yhat_probs)
    # predict crisp classes for test set deprecated
    #printMetrics(y_test,yhat_probs)
    test = list()

    xx = yhat_probs.round()

    y_test2 = pd.DataFrame(data=xx,columns=['awake','asleep']) 
    y_test_label_label = pd.DataFrame(data=y_test_label,columns=['awake','asleep']) 
    
    # generate time metrics
    current_time2 = datetime.datetime.now()
    time_stamp2 = current_time2.timestamp()
    processing_time_s = (time_stamp2-time_stamp)
    # generate general metrics
    rowData = [i,current_time,current_time2,processing_time_s,(processing_time_s)/60]

    print('')
    print('awake')    
    res,resA = printMetrics(y_test_label_label['awake'],y_test2['awake'])
    test.append(res)
   
    #columns = ['NN_type','units','epochs','batch_size','max_iterations',''Users',
    #            round_iteration','start_time','end_time','round_time_s','round_time_m',
    #           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
    #           'TP','FP','FN','TN']
    # new data
    classData = np.concatenate((['awake'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    roundData.append(classData)
    
    print('')
    print('asleep')
    res,resA = printMetrics(y_test_label_label['asleep'],y_test2['asleep'])
    test.append(res)
    # new data
    classData = np.concatenate((['asleep'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    roundData.append(classData)
    print('')
    print('Global')
    resA = showGlobalMetrics(test) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    # new data
    classData = np.concatenate((['avg'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    roundData.append(classData)
    print('')
    print(i,"End timestamp:", time_stamp2,current_time2)
    print('')
    print('')


# In[ ]:


dataMetrics = pd.DataFrame(data=roundData,columns=columns) 

dataMetrics


# In[ ]:


dataMetrics.to_csv(outputMetricFile, sep=',', encoding='utf-8', index=False)


# In[ ]:




