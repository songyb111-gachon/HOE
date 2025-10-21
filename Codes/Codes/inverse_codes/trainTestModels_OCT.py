import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils

import inputOutput
import inputOutput_OCT
import deepModels
import calculateLoss
import trainTestModels
############################################################################################################
def returnAuxInfo(taskNo):
    outputTypes = ['MC']
    taskWeights = [1.0]
    outputChannelNos = [2]    
    
    if taskNo >= 2:
        for i in range(1, taskNo):
            outputTypes.append('R')
            taskWeights.append(1.0)
            outputChannelNos.append(1)
            
    return [outputChannelNos, outputTypes, taskWeights]
############################################################################################################
def taskLists4UNet(imageNames, inputPath, outputPath, coordinate):
    setInputList = []
    setOutputList = []

    [setInputs, setOutputs, setNames, outputTypes] = inputOutput_OCT.readOneOCTDataset(imageNames, inputPath, outputPath, coordinate)
    tSize, im_height, im_width, taskNo = setOutputs.shape
    setInputList.append(setInputs)
    for i in range(taskNo):
        if outputTypes[i] != ".png":
            currW = calculateLoss.calculateLossWeightsForOneDataset(setOutputs[:, :, :, i], 'same')
            currOut = setOutputs[:, :, :, i].reshape(tSize, im_height ,im_width, 1)
        else:
            currW = calculateLoss.calculateLossWeightsForOneDataset(setOutputs[:, :, :, i], 'class-weighted')    
            currOut = inputOutput.createCategoricalOutput(setOutputs[:, :, :, i], 'MC')
            
        setInputList.append(currW)
        setOutputList.append(currOut)
        
    return [setInputList, setOutputList, taskNo]
############################################################################################################
def trainUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
              layerNum, noOfFeatures, dropoutRate, learningRate, coordinate):
    [trInputList, trOutputList, taskNo] = taskLists4UNet(trImageNames, trInputPath, trOutputPath, coordinate)
    [valInputList, valOutputList, taskNo] = taskLists4UNet(valImageNames, valInputPath, valOutputPath, coordinate)
    [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
    
    trainTestModels.trainModel(modelType, modelFile, trInputList, trOutputList, valInputList, valOutputList, taskWeights, 
                               layerNum, noOfFeatures, dropoutRate, outputChannelNos, outputTypes, learningRate)
############################################################################################################
def testUnet(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, coordinate):
    [tsInputs, tsOutputs, tsNames, outputTypes] = inputOutput_OCT.readOneOCTDataset(tsImageNames, tsInputPath, tsOutputPath, coordinate)
    taskNo = tsOutputs.shape[3]
    [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
    
    model = trainTestModels.loadModel(modelType, modelFile, tsInputs, taskWeights, noOfFeatures, dropoutRate, layerNum,
                                      outputChannelNos, outputTypes)
    probs = trainTestModels.testModel(model, tsInputs, len(outputTypes))
    return [probs, tsNames, taskNo]
############################################################################################################
