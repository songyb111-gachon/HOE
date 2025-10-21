import tensorflow as tf
from keras import backend as K
import numpy as np

from keras import optimizers, losses
from keras.utils import np_utils
from scipy.ndimage.morphology import distance_transform_edt

############################################################################################################
def weighted_binary_crossentropy(y_weight):
    def loss(y_true, y_pred):
        bce = losses.binary_crossentropy(y_true, y_pred)
        loss = (y_weight * bce)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_categorical_crossentropy(y_weight):
    def loss(y_true, y_pred):
        cce = losses.categorical_crossentropy(y_true, y_pred)
        loss = (y_weight * cce)
        return K.mean(loss)
    return loss
############################################################################################################
def cosine_similarity(y_true, y_pred):
    
    y_true = K.l2_normalize(y_true[:,:,0])
    y_pred = K.l2_normalize(y_pred[:,:,0])
    return - K.sum(y_true * y_pred)
    
def categorical_crossentropy(output1, output2):
    def loss(y_true, y_pred):
        cce = losses.categorical_crossentropy(y_true, y_pred)        
        cos_sim_loss = cosine_similarity(output1, output2)
        return cce + cos_sim_loss
    return loss
############################################################################################################
def weighted_mse(y_weight):
    def loss(y_true, y_pred):
        mse = objectives.mean_squared_error(y_true, y_pred)
        loss = (y_weight * mse)
        return K.mean(loss)
    return loss
############################################################################################################
def calculateUnetBoundaryWeightMapOneImage(gold, wc, w0 = 10, sigma = 5):
    
    foreground = (gold > 0)
    background = gold == 0
    goldIds = sorted(np.unique(gold))[1:]
    if len(goldIds) >= 1:
        distances = np.zeros((foreground.shape[0], foreground.shape[1], len(goldIds)))
        for i, label_id in enumerate(goldIds):
            distances[:, :, i] = distance_transform_edt(gold != label_id)
        distances = np.sort(distances, axis=2)

        d1 = distances[:, :, 0]        
        if len(goldIds) == 1:    
            d2 = distances[:, :, 0]
        else:
            d2 = distances[:, :, 1]
        w = w0 * np.exp(-1/2 * ((d1 + d2) / sigma)**2) * background
    else:
        w = np.zeros_like(foreground)
        
    if wc:
        class_weights = np.zeros((gold.shape[0], gold.shape[1]))
        for k, v in wc.items():
            class_weights[foreground == k] = v
        w = w + class_weights

    w = (foreground.shape[0] * foreground.shape[1] * w) / w.sum()
    return w
############################################################################################################
def unetDistanceWeightMaps(golds, classWeighted):
    imageNo = golds.shape[0]
    inputHeight = golds.shape[1]
    inputWidth = golds.shape[2]

    imageWeights = np.zeros((imageNo, inputHeight, inputWidth))
    for i in range(imageNo):
        if classWeighted:
            class1Perc = golds[i].sum() / (inputHeight * inputWidth)
            wc = {0:class1Perc, 1:(1-class1Perc)}
        else:
            wc = {0:1, 1:1}
        
        imageWeights[i] = calculateUnetBoundaryWeightMapOneImage(golds[i], wc)
        
    return imageWeights
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
def sameWeightsForAll(masks):
    imageNo = masks.shape[0]
    inputHeight = masks.shape[1]
    inputWidth = masks.shape[2]

    imageWeights = np.ones((imageNo, inputHeight, inputWidth))
    return imageWeights
############################################################################################################
def calculateLossWeightsForOneDataset(setOutputs, lossType):
    if lossType == 'same':
        setWeights = sameWeightsForAll(setOutputs)
    elif lossType == 'class-weighted':
        setWeights = classWeightMaps(setOutputs)
    elif lossType == 'unet-distance':
        setWeights = unetDistanceWeightMaps(setOutputs, False)
    elif lossType == 'unet-distance-class-weighted':
        setWeights = unetDistanceWeightMaps(setOutputs, True)

    return setWeights
############################################################################################################
