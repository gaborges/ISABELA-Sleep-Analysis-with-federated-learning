#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
import tensorflow_addons as tfa
import pandas as pd
# Bibliotecas Auxiliares
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print(tf.__version__)


# In[12]:


# configs
EPOCHS = 100
BATCH_SIZE = 32

baseFolder = "../data_2019_processed/"


# In[13]:


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
    
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['cohen_kappa_score'] = kappa
    results['roc_auc_score'] = auc
    return results

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


# In[14]:


#X_train = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized.csv")
X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized.csv")
#X_train_under = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized_balanced_undersample.csv")
#X_train_over = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized_balanced_oversample.csv")
X_train_over = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized_balanced_oversample_smote.csv")
#X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized_balanced_undersample.csv")
X_train = X_train_over
#AA = pd.read_csv(baseFolder+"allData-classification-numeric-normalized.csv")

#X_train, X_test = train_test_split(AA,test_size=0.25)
#X_train = pd.read_csv("train_temp.csv")
#X_test = pd.read_csv("test_temp.csv")


# In[15]:


print(X_train.info())
X_train


# In[16]:


# create two classes based on the single class

# transform output to one_hot_encoding for the testing dataset
X_test = transform_output_nominal_class_into_one_hot_encoding(X_test)

# transform output to one_hot_encoding for the testing dataset
X_train = transform_output_nominal_class_into_one_hot_encoding(X_train)

# transform output to one_hot_encoding for the testing dataset
#X_train_under = transform_output_nominal_class_into_one_hot_encoding(X_train_under)

# transform output to one_hot_encoding for the testing dataset
X_train_over = transform_output_nominal_class_into_one_hot_encoding(X_train_over)


# In[20]:


# activity	location	timestamp	day_of_week	light	phone_lock	proximity	sound	time_to_next_alarm	minutes_day
inputFeatures = ["activity","location","day_of_week","light","phone_lock","proximity","sound","time_to_next_alarm", "minutes_day"]
#inputFeatures = ["activity","location","day_of_week","light","phone_lock","proximity","sound", "minutes_day"]

outputClasses = ["awake","asleep"]



def create_keras_model(inputFeatures,outputClasses):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(len(inputFeatures),)),
        tf.keras.layers.Dense(len(inputFeatures)),
        tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(12, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(len(outputClasses), activation='softmax')
    ])


# In[21]:


print("--------------")
print("--------------")
print("Initializing Oversample SMOTE test")
print("--------------")
print("Batch Size: ",BATCH_SIZE)
print("Epochs: ",EPOCHS)
print("Input/Features:",len(inputFeatures))
print("Outputs:",len(outputClasses))
print("--------------")

# selects the data to train and test
X_train_data = pd.DataFrame(data=X_train_over,columns=inputFeatures)
y_train_data = pd.DataFrame(data=X_train_over,columns=outputClasses)
# selec test dataset (fixed to all)
X_test_data = pd.DataFrame(data=X_test,columns=inputFeatures)
y_test_data = pd.DataFrame(data=X_test,columns=outputClasses)

# instanciates the model
model = create_keras_model(inputFeatures,outputClasses)

# compiles the model
model.compile(optimizer='adam',
             # loss='binary_crossentropy',loss='categorical_crossentropy',
              #loss='binary_crossentropy',  sparse_categorical_crossentropy         
              loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
# 
model.fit(X_train_data, y_train_data, epochs=EPOCHS,batch_size=BATCH_SIZE, verbose=0)

results = model.evaluate(X_test_data, y_test_data, return_dict=True)

print(results)


yhat_probs = model.predict(X_test_data)
# dataset
probss = pd.DataFrame(data=yhat_probs,columns=['awake','asleep'])

test = list()
print('')
print('awake')
res = printMetrics(y_test_data['awake'],probss['awake'])
test.append(res)
print('')
print('asleep')
res = printMetrics(y_test_data['asleep'],probss['asleep'])
test.append(res)
print('')
print('Global')
showGlobalMetrics(test)


# In[ ]:
