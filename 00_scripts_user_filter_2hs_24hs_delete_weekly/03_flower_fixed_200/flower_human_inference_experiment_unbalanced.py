#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from multiprocessing import Process
import gc

import tensorflow as tf
import sklearn
import time

import numpy as np
import pandas as pd
from pandas import DataFrame

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

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# In[2]:


# argumentos
n = len(sys.argv)
print("Total arguments passed:", n)
iteracoes = 0
cycle_index = 1
finalIterations = 0
if(n > 0):
    for value in sys.argv:
        print("arg:", value)
        if("iterations=" in value):
            try:
                iteracoes = int(value.replace("iterations=",""))
            except:
                print("no")
        
        if("cycle=" in value):
            try:
                cycle_index = int(value.replace("cycle=",""))
            except:
                print("no")
print("iteracoes:",iteracoes)      
print("cycle:",cycle_index)

# In[3]:


# input folder
#inputFolders = "../02-transformed-data-new-testes/dados2019/"
inputFolderPath = "../data_2019_processed/"
inputFolderPath = "../data_2019_processed/" # not filtered

# General configuration
NUMBER_OF_ITERATIONS_FINAL = 200
    
NUM_EPOCHS = 1
BATCH_SIZE = 32
VERBOSE = 0


# usado para checkpoints
if(iteracoes > 0):
    NUMBER_OF_ITERATIONS_FINAL = iteracoes
    
NUMBER_OF_ITERATIONS = NUMBER_OF_ITERATIONS_FINAL


# output folder
outputFolder = "result_unbalanced_epoch_"+str(NUM_EPOCHS)+"_rounds_"+str(NUMBER_OF_ITERATIONS_FINAL)+"_cycle_"+str(cycle_index)
#outputFolder = "test_checkpoint"
checkPointFolder = outputFolder+"/checkpoints"
iferredCycleDataFolder = outputFolder+"/inferred_datasets"

# train file name modifier
fileSufixTrain = "_transformed" # _transformed_smote for smote

#fl.common.logger.configure(identifier="myFlowerExperiment", filename="log_"+outputFolder+".txt")


# In[4]:


print("Checking whether checkpoint exist")
print(checkPointFolder)
isExist = os.path.exists(checkPointFolder)
if not isExist:
    # Create a new directory because it does not exist
    print("There is no checkpoint available to continue the process!")
    sys.exit("There is no checkpoint available to continue the process!")


# In[5]:


print("Checking whether the data folder exists or not")
isExist = os.path.exists(iferredCycleDataFolder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(iferredCycleDataFolder)
    print("The new directory is created! ",iferredCycleDataFolder)
else:
    print("The directory exists!")


# In[6]:


# selected features
inputFeatures = ["activity","location","day_of_week","light","phone_lock","proximity","sound","time_to_next_alarm", "minutes_day"]
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]


# In[7]:


# client datasets used on the training process (75% of data)
trainFolders =  ['0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
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
                #'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', # does not have asleep data
                'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
                'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
                'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
                'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
                'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
                'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
                'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
                #'PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc', # does not have asleep data
                'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
                #'rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw', 
                #'RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI', 
                'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4']
                #'VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is', 
                #'Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw', 
                #'XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA', 
                #'YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw', 
                #'ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM', 
                #'ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']
            
# client datasets used on the training process (25% of data)
testFolders =  [#'0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
                #'0tdmm6rwW3KquQ73ATYYJ5JkpMtvbppJ0VzA2GExdA', 
                #'2cyV53lVyUtlMj0BRwilEWtYJwUiviYoL48cZBPBq0', 
                #'2J22RukYnEbKTk7t+iUVDBkorcyL5NKN6TrLe89ys', 
                #['5FLZBTVAPwdq9QezHE2sVCJIs7p+r6mCemA2gp9jATk'], #does not have the file
                #'7EYF5I04EVqisUJCVNHlqn77UAuOmwL2Dahxd3cA', 
                #'a9Qgj8ENWrHvl9QqlXcIPKmyGMKgbfHk9Dbqon1HQP4', 
                #'ae4JJBZDycEcY8McJF+3BxyvZ1619y03BNdCxzpZTc', 
                #'Ch3u5Oaz96VSrQbf0z31X6jEIbeIekkC0mwPzCdeJ1U', 
                #'CH8f0yZkZL13zWuE9ks1CkVJRVrr+jsGdUXHrZ6YeA', 
                #'DHO1K4jgiwZJOfQTrxvKE2vn7hkjamigroGD5IaeRc', 
                #'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', # does not have asleep data
                #'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
                #'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
                #'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
                #'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
                #'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
                #'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
                #'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
                #'PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc', # does not have asleep data
                #'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
                'rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw', 
                'RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI', 
                #'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4'] 
                'VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is', 
                'Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw', 
                'XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA', 
                'YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw', 
                'ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM', 
                'ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']


# In[8]:


def generateMetrics(y_test,yhat_probs):
    # predict crisp classes for test set deprecated
    #yhat_classes = model.predict_classes(X_test, verbose=0)
    #yhat_classes = np.argmax(yhat_probs,axis=1)
    yhat_classes = yhat_probs.round()
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    #print(matrix)
    
    array = []
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['cohen_kappa_score'] = kappa
    results['roc_auc_score'] = auc
    results['matrix'] = ("[[ " +str(matrix[0][0]) + " " +str(matrix[0][1]) +"][ " +str(matrix[1][0]) + " " + str(matrix[1][1]) +"]]") # array.append(np.array(matrix,dtype=object))
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
    array.append("[[ " +str(matrix[0][0]) + " " +str(matrix[0][1]) +"][ " +str(matrix[1][0]) + " " + str(matrix[1][1]) +"]]") # array.append(np.array(matrix,dtype=object))
    array.append(matrix[0][0]) # TP
    array.append(matrix[0][1]) # FP
    array.append(matrix[1][0]) # FN
    array.append(matrix[1][1]) # TN
    
    return results, array

# y_test     = Array with real values
# yhat_probs = Array with predicted values
def printMetrics(y_test,yhat_probs):
    # generate metrics
    results, array= generateMetrics(y_test,yhat_probs)

    # accuracy: (tp + tn) / (p + n)
    accuracy = results['accuracy']
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = results['precision']
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = results['recall'] 
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = results['f1_score']
    print('F1 score: %f' % f1)
    # kappa
    kappa = results['cohen_kappa_score']
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = results['roc_auc_score']
    print('ROC AUC: %f' % auc)
    # confusion matrix
    print("\Confusion Matrix")
    matrix = results['matrix']
    print(matrix)
    
    return results, array

def generateGlobalMetrics(metrics):
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
    
    return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score]

def showGlobalMetrics(metrics):
    res = generateGlobalMetrics(metrics)
    
    accuracy = res[0]
    precision = res[1]
    recall = res[2]
    f1_score = res[3]
    cohen_kappa_score = res[4]
    roc_auc_score = res[5]
    
    #show:\
    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1_score: ",f1_score)
    print("cohen_kappa_score: ",cohen_kappa_score)
    print("roc_auc_score: ",roc_auc_score)
    
    return res


# In[9]:


# take the list of directories and concat them
def loadDataFromFolders(foldersToLoad,inputFolders,fileType = "_transformed"):
    print(len(foldersToLoad), "datasets")
    for i in range(0,len(foldersToLoad)):
        currentFolder = foldersToLoad[i]
        print(i , "-", currentFolder,inputFolders+"student_"+currentFolder+fileType+".csv")
        #print(trainingDataSet[i])
        if(i == 0):
            temp_data = pd.read_csv(inputFolders+"student_"+currentFolder+fileType+".csv")
        else:
            dataset = pd.read_csv(inputFolders+"student_"+currentFolder+fileType+".csv")
            temp_data = pd.concat([temp_data, dataset])
    # return the dataset        
    return temp_data

# take the list of directories and concat them
def loadDataFromFoldersOnList(foldersToLoad,inputFolders,fileType = "_transformed"):
    clientList = []
    print(len(foldersToLoad), "datasets")
    for i in range(0,len(foldersToLoad)):
        currentFolder = foldersToLoad[i]
        print(i , "-", currentFolder,inputFolders+"student_"+currentFolder+fileType+".csv")
        #print(trainingDataSet[i])
        temp_data = pd.read_csv(inputFolders+"student_"+currentFolder+fileType+".csv")
        print("Adding to the list: ", temp_data.shape)
        clientList.append(temp_data)
    # return the dataset        
    return clientList


# Load datasets

# In[10]:


print("Preparing test data")
 
# test data comprising 25% of the data. It must be fixed to all models being evaluated
#X_test  = pd.read_csv(inputFolders+"test/allData-classification-numeric-normalized.csv")
X_test = loadDataFromFolders(testFolders,inputFolderPath,fileSufixTrain)

print()
# undestand the dataset by looking on their infos
print(X_test.info())

X_test


# In[11]:


print("Preparing X_train data")
# load cliend data
clientList = loadDataFromFoldersOnList(trainFolders,inputFolderPath,"") # original data witout filter
        
NUMBER_OF_CLIENTS = len(clientList)
print("Total",(len(clientList)))


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
    clientList[i] = transform_output_nominal_class_into_one_hot_encoding(clientList[i])
    #print (clientList[i])
    

X_test.info()


# In[ ]:





# In[13]:


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


# In[14]:


print("Prepering the test dataset")
# selects the data to train and test
X_test_data = X_test[inputFeatures]
y_test_label = X_test[outputClasses]

# transtorm data to tensor slices
#client_test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data.values, y_test_label.values))

#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True)
#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)

#print(client_test_dataset.element_spec)
#client_test_dataset


# In[ ]:





# --
# --
# Load model from checkpoint
# --
# --

# In[15]:


print("creating model")

def create_keras_model():
    return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(9,)),
      #tf.keras.layers.Dense(9, activation=tf.keras.activations.relu), 
      tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),
      tf.keras.layers.Dense(8, activation=tf.keras.activations.relu),
      tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
      #tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])

keras_model = create_keras_model()
#keras_model.summary()
keras_model.summary()


# In[16]:


# comment for the future: I should load the models thinking on:
# 1 - best global accuracy
# 2 - best model for each device (I should save the best model for each device)
# 3 - maybe I have to try use the metric roc

print("Loading checkpoint model",checkPointFolder+"/round-*")
list_of_files = [fname for fname in glob.glob(checkPointFolder+"/round-*")]
last_round_checkpoint = -1
latest_round_file = None
model_check_point = None
filename_h5 = None
filename_np = None

if len(list_of_files) > 0:
    valuesTempString = [value.replace(outputFolder+"/checkpoints/round-","").replace("-weights.h5","").replace("-weights.npz","") for value in list_of_files]
    if(len(valuesTempString) > 0):
        last_round_checkpoint = max([int(value) for value in valuesTempString])
        print("Loading pre-trained model from: ", last_round_checkpoint)
        # load the name
        print("Last round: ",last_round_checkpoint)
    
        filename_h5 = checkPointFolder+"/round-"+str(last_round_checkpoint)+"-weights.h5"
        filename_np = checkPointFolder+"/round-"+str(last_round_checkpoint)+"-weights.npz"
else:
    print("No checkpoint file found")
    sys.exit("No checkpoint file found")


# In[17]:


# load model
# load checkpoint
print("load checkpoint")
keras_model.load_weights(filename_h5)


# In[18]:


# for each data infer the output and save the new value
print(len(trainFolders), "datasets")
for i in range(0,len(trainFolders)): 
    currentFolder = trainFolders[i]
    # selects the data to train and test
    X_train_data = clientList[i][inputFeatures]
    y_train_label = clientList[i][outputClasses]

    inputData = X_train_data
    outputFileName = iferredCycleDataFolder+"/student_"+currentFolder+".csv"
    print(i , "-", outputFileName)
    #print(trainingDataSet[i])
    
    yhat_probs = keras_model.predict(inputData,verbose=VERBOSE)
    
    # as we deal with a classification problem with one hot encoding, we must round the values to 0 and 1.
    yhat_probs_rounded = yhat_probs.round()
    
    # create a dataframe with the predicted data
    y_predicted_df = pd.DataFrame(data=yhat_probs_rounded,columns=['awake','asleep']) 

    outputData = pd.concat([inputData,y_predicted_df], axis=1)
    #inputData['awake'] = y_predicted_df['awake']
    #inputData['asleep'] = y_predicted_df['asleep']

    outputData.to_csv(outputFileName, sep=',', encoding='utf-8', index=False)


# In[19]:


outputData


# In[20]:


# compare data
# for each data infer the output and save the new value
datasetsMixed = []
print(len(trainFolders), "datasets")
for i in range(0,len(trainFolders)): 
    currentFolder = trainFolders[i]
    # selects the data to train and test
    X_train_data = clientList[i][inputFeatures]
    y_train_label = clientList[i][outputClasses]

    inputData = X_train_data
    outputFileName = iferredCycleDataFolder+"/student_"+currentFolder+".csv"
    print(i , "-", outputFileName)
    #print(trainingDataSet[i])
    
    yhat_probs = keras_model.predict(inputData,verbose=VERBOSE)
    
    # as we deal with a classification problem with one hot encoding, we must round the values to 0 and 1.
    yhat_probs_rounded = yhat_probs.round()
    
    # create a dataframe with the predicted data
    y_predicted_df = pd.DataFrame(data=yhat_probs_rounded,columns=['awake_pred','asleep_pred']) 

    outputData = pd.concat([inputData,y_train_label,y_predicted_df], axis=1)
    #inputData['awake'] = y_predicted_df['awake']
    #inputData['asleep'] = y_predicted_df['asleep']

    datasetsMixed.append(outputData)


# In[21]:


datasetsMixed[0]


# In[36]:


metrics = None
metricsa = None
metricsAll = []
predOutputClasses = ["awake_pred","asleep_pred"]
for i in range(0,len(datasetsMixed)): 
    currentOne = datasetsMixed[i]
    currentOne = transform_data_type(currentOne)
    realLabel = currentOne[outputClasses]     # ['awake', 'asleep']
    predLabel = currentOne[predOutputClasses]
    
    
    metrics,metricsa = generateMetrics(realLabel["awake"],predLabel["awake_pred"])
    metricsAll.append(metrics)
    print()
    print()
    printMetrics(realLabel["awake"],predLabel["awake_pred"])
    #printMetrics(realLabel["asleep"],predLabel["asleep_pred"])
    print()
    print()


# In[35]:


metrics


# In[37]:


metricsAll


# In[82]:


currentOne


# In[79]:


datasetsMixed[i][predOutputClasses]


# In[83]:


realLabel


# In[84]:


predLabel


# In[25]:


#np.array(attribute1).T


# In[26]:


#attribute1


# In[27]:


#np.array(attribute1)


# In[41]:


import matplotlib.pyplot as plt
import numpy as np

# Your array containing the datasets
# Let's assume it's named 'data'
# Each element of 'data' represents a row and the attributes are the columns
data = datasetsMixed  # Your array with 18 datasets
selectedAttributes = ['awake','awake_pred']

# Determine the number of datasets and attributes
num_datasets = len(data)
num_attributes = len(selectedAttributes)  # Assuming all datasets have the same number of attributes

# Calculate the number of rows and columns for subplots
num_rows = num_datasets
num_cols = num_attributes  # Two subplots per attribute

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))

# Plot each dataset in its own subplot
for i, dataset in enumerate(data):
    for j, attribute in enumerate(selectedAttributes):
        ax = axs[i, j]
        if(j % 2 == 1):
            ax.plot(dataset[attribute], label=attribute, linestyle='--', marker='o', color='red')
        else:
            ax.plot(dataset[attribute], label=attribute, linestyle='--', marker='o')
        ax.set_title(f'Dataset {i+1}, Attribute: {attribute}, Accuracy: {round(metricsAll[i]["accuracy"], 2)} ')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import numpy as np

# Your array containing the datasets
# Let's assume it's named 'data'
# Each element of 'data' represents a row and the attributes are the columns
data = datasetsMixed  # Your array with 18 datasets
selectedAttributes = ['awake', 'awake_pred']

# Determine the number of datasets and attributes
num_datasets = len(data)
num_attributes = len(selectedAttributes)  # Assuming all datasets have the same number of attributes

# Calculate the number of rows and columns for subplots
num_rows = num_datasets
num_cols = num_attributes + 1  # Three subplots per attribute

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))

# Plot each dataset in its own subplot
for i, dataset in enumerate(data):
    for j, attribute in enumerate(selectedAttributes):
        ax = axs[i, j]
        ax.plot(dataset[attribute], label=attribute, linestyle='--', marker='o')
        ax.set_title(f'Dataset {i+1}, Attribute: {attribute}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()

    # Calculate and plot the difference between the selected attributes
    diff = dataset[selectedAttributes[1]] - dataset[selectedAttributes[0]]
    ax_diff = axs[i, num_attributes]  # Access the third subplot in the row
    ax_diff.plot(diff, label='Difference', linestyle='-', marker='s', color='red')
    ax_diff.set_title(f'Dataset {i+1}, Difference')
    ax_diff.set_xlabel('Index')
    ax_diff.set_ylabel('Difference')
    ax_diff.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[ ]:




