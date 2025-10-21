import os
import numpy as np
import sys, getopt
from keras.utils import np_utils

import inputOutput
import inputOutput_OCT
import trainTestModels_OCT
############################################################################################################          
def callUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
             tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, tsResultPath, trainStr, channel_name, runNo):
    if trainStr == 'tr':
        trainTestModels_OCT.trainUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, channel_name)
        
        trainTestModels_OCT.testUnet_with_mat_file(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, channel_name, tsResultPath, runNo)
    elif trainStr == 'ts':  
        trainTestModels_OCT.testUnet_with_mat_file(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, channel_name, tsResultPath, runNo)
    elif trainStr == 'ts_lens':  
        trainTestModels_OCT.testUnet_with_mat_file_lens(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, channel_name, tsResultPath, runNo)
############################################################################################################
def OCTSegmentation(networkType, foldNo, runNo, trainStr, learningRate, serverStr, channel_name):
    if serverStr == 'KU':
        dataPath = '/userfiles/cgunduz/datasets/OCT'
        savePathPrefix = '/kuacc/users/cgunduz/'
    elif serverStr == 'apollo':
        dataPath = '../results_matlab/'
        savePathPrefix = './'
    
    trInputPath = dataPath
    valInputPath = dataPath
    tsInputPath = dataPath
    
    outputPathtr = ''
    outputPathval = ''
    
    fdmapPath = dataPath + '/fdmaps_center/'
    imagePostfix = '.png'

    trImageNames = ''
    valImageNames = ''
    tsImageNames = ''
    
    trOutputPath = ''
    valOutputPath = ''
    tsOutputPath = ''
    
    layerNum = 5
    noOfFeatures = [16, 32, 64, 128, 256, 512]
    dropoutRate = 0.2
    
    [modelFile, tsResultPath] = inputOutput.generateOutputNames(savePathPrefix, networkType, 'oct', layerNum, 
                                                                noOfFeatures[0], foldNo, runNo)
    callUnet('unet', modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
             tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, learningRate, tsResultPath, trainStr, channel_name, runNo)
############################################################################################################
