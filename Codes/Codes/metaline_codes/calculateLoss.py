import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.layers import Layer, Dense, Activation, Flatten, Reshape, Permute, Lambda
from tensorflow.keras import  metrics,optimizers, losses


############################################################################################################
def weighted_binary_crossentropy(y_true, y_pred, y_weight):
    bce = losses.binary_crossentropy(y_true, y_pred)
    loss = bce * y_weight
    return K.mean(loss)
############################################################################################################
def weighted_categorical_crossentropy(y_weight):
    def loss(y_true, y_pred):        
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = (y_weight * cce)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_mse(y_weight):
    def loss(y_true, y_pred): 
        mse = losses.mean_squared_error(y_true, y_pred)
        loss = (y_weight * mse)
        return K.mean(loss)
    return loss
############################################################################################################
def calculateClassWeights(gold):
    totalPixels = gold.shape[0] * gold.shape[1]
    classNo = int(gold.max() + 1)
    classCounts = np.zeros(classNo)
    classWeights = np.zeros(classNo)
    for i in range(classNo):
        classCounts[i] = (gold == i).sum()
        classWeights[i] = classCounts[i] / totalPixels

    total = 0
    for i in range(classNo):
        if classWeights[i] > 0:
            classWeights[i] = 1 / classWeights[i]
            total += classWeights[i]
            
    for i in range(classNo):
        classWeights[i] /= total
        
    return classWeights
############################################################################################################        
def calculateClassWeightMapOneImage(gold):
    classWeights = calculateClassWeights(gold)

    inputHeight = gold.shape[0]
    inputWidth = gold.shape[1]
    
    lossWeightMap = np.zeros((inputHeight, inputWidth))
    for i in range(inputHeight):
        for j in range(inputWidth):
            lossWeightMap[i][j] = classWeights[int(gold[i][j])]
            
    lossWeightMap = (inputHeight * inputWidth * lossWeightMap) / lossWeightMap.sum()
            
    return lossWeightMap
############################################################################################################
def classWeightMaps(golds):
    imageNo = golds.shape[0]
    inputHeight = golds.shape[1]
    inputWidth = golds.shape[2]

    imageWeights = np.zeros((imageNo, inputHeight, inputWidth))
    for i in range(imageNo):
        imageWeights[i] = calculateClassWeightMapOneImage(golds[i])
        
    return imageWeights
############################################################################################################
def trValWeights(trOutputs, valOutputs, lossType):
    if lossType == 'same':
        trWeights = np.ones((trOutputs.shape[0], trOutputs.shape[1], trOutputs.shape[2]))
        valWeights = np.ones((valOutputs.shape[0], valOutputs.shape[1], valOutputs.shape[2]))
    elif lossType == 'class-weighted':
        trWeights = classWeightMaps(trOutputs)
        valWeights = classWeightMaps(valOutputs)
    return [trWeights, valWeights]
############################################################################################################
