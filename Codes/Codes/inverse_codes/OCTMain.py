import os
import numpy as np
import sys, getopt
from keras.utils import np_utils

import inputOutput
import inputOutput_OCT
import trainTestModels_OCT
############################################################################################################          
def callUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
             tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, tsResultPath, trainStr, coordinate):
    if trainStr == 'tr':
        trainTestModels_OCT.trainUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, coordinate)
        
        [predictions, tsNames, taskNo] = trainTestModels_OCT.testUnet(modelType, modelFile, tsImageNames, tsInputPath, 
                                                                      tsOutputPath, layerNum, noOfFeatures, dropoutRate, coordinate)
        if (taskNo == 1):
            inputOutput.saveAsDouble(tsResultPath, tsNames, predictions[:, :, :, 1], '_pred')
        else:
            inputOutput.saveAsDouble(tsResultPath, tsNames, predictions[0][:, :, :, 1], '_pred')
    else:
        [predictions, tsNames, taskNo] = trainTestModels_OCT.testUnet(modelType, modelFile, tsImageNames, tsInputPath, 
                                                                      tsOutputPath, layerNum, noOfFeatures, dropoutRate, coordinate)
        if (taskNo == 1):
            inputOutput.saveAsDouble(tsResultPath, tsNames, predictions[:, :, :, 1], '_pred')
        else:
            inputOutput.saveAsDouble(tsResultPath, tsNames, predictions[0][:, :, :, 1], '_pred')
############################################################################################################
def OCTSegmentation(networkType, foldNo, runNo, trainStr, learningRate, serverStr, coordinate):
    if serverStr == 'KU':
        dataPath = '/userfiles/cgunduz/datasets/OCT'
        savePathPrefix = '/kuacc/users/cgunduz/'
    elif serverStr == 'apollo':
        dataPath = '../data'
        savePathPrefix = '../inverse_problem/'
    
    trInputPath = dataPath + '/bilge_mini_data_circular/training/outputs/'
    valInputPath = dataPath + '/bilge_mini_data_circular/validation/outputs/'

    tsInputPath = dataPath + '/test_bilge_cyl/outputs/'
    outputPathtr = dataPath + '/bilge_mini_data_circular/training/inputs/'
    outputPathval = dataPath + '/bilge_mini_data_circular/validation/inputs/'
    imagePostfix = '_x1.txt'

    trImageNames = inputOutput_OCT.listAllOCTFiles(trInputPath, imagePostfix)
    valImageNames = inputOutput_OCT.listAllOCTFiles(valInputPath, imagePostfix)
    tsImageNames = inputOutput_OCT.listAllOCTFiles(tsInputPath, imagePostfix)
    
    trOutputPath = outputPathtr
    valOutputPath = outputPathval
    tsOutputPath = ''
    
    layerNum = 5
    noOfFeatures = [64, 128, 256, 512, 1024, 2048]
    dropoutRate = 0.2
    
    [modelFile, tsResultPath] = inputOutput.generateOutputNames(savePathPrefix, networkType, 'oct', layerNum, 
                                                                noOfFeatures[0], foldNo, runNo)
    callUnet('unet', modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
             tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, tsResultPath, trainStr, coordinate)
############################################################################################################
