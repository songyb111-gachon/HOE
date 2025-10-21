import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
from tensorflow.python.framework.ops import disable_eager_execution

import inputOutput
import deepModels
import calculateLoss
############################################################################################################
def createCallbacks(modelFile, earlyStoppingPatience):
    checkpoint = ModelCheckpoint(modelFile, monitor = 'val_loss', verbose = 1, save_best_only = True, 
                                 save_weights_only = True, mode = 'auto', period = 1)
    earlystopping = EarlyStopping(patience = earlyStoppingPatience, monitor = 'val_loss', verbose = 1, 
                                 restore_best_weights = True)
    return [checkpoint, earlystopping]
############################################################################################################
def plotConvergencePlots(hist, modelFile):
#    fig = plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plotFile = modelFile[:-5] + '.png'
    plt.savefig(plotFile)
#    plt.close(fig)
############################################################################################################
def taskLists4UNet(imageNames, inputPath, inputExtension, isGray, outputPaths, outputExtensions, outputTypes, lossTypes):
    setInputList = []
    setOutputList = []
    [setInputs, setOutputs, setNames, outputTypes] = inputOutput.readOneDataset(imageNames, inputPath, inputExtension, isGray,
                                                       outputPaths, outputExtensions, outputTypes)
    
    tSize, im_height, im_width, taskNo = setOutputs.shape
    setInputList.append(setInputs)
    for i in range(taskNo):
        if outputTypes[i] != "hfl":
            currW = calculateLoss.calculateLossWeightsForOneDataset(setOutputs[:, :, :, i], 'same')
            currOut = setOutputs[:, :, :, i].reshape(tSize, im_height ,im_width, 1)
        else:
            currW = calculateLoss.calculateLossWeightsForOneDataset(setOutputs[:, :, :, i], 'class-weighted')
            currOut = inputOutput.createCategoricalOutput(setOutputs[:, :, :, i], 'MC')
            
        setInputList.append(currW)
        setOutputList.append(currOut)
        
    return [setInputList, setOutputList]
############################################################################################################
def trainModel(modelType, modelFile, trainInputList, trainOutputList, validInputList, validOutputList, taskWeights,
               layerNum, noOfFeatures, dropoutRate, outputChannelNos = [], outputTypes = [], 
               lr = 0.001, earlyStoppingPatience = 50, batchSize = 1, maxEpoch = 500):
    inputHeight = trainInputList[0].shape[1]
    inputWidth = trainInputList[0].shape[2]
    channelNo = trainInputList[0].shape[3]
    disable_eager_execution()
    
    if modelType == 'unet':
        model = deepModels.unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, layerNum, 
                                noOfFeatures, dropoutRate, taskWeights, lr)
    elif modelType == 'simple-cascade' or modelType == 'input-cascade':
        model = deepModels.cascaded(modelType, inputHeight, inputWidth, channelNo, outputChannelNos[0], outputChannelNos[1],
                                layerNum, noOfFeatures, dropoutRate, taskWeights, lr)    
    model.summary()
    hist = model.fit(x = trainInputList, y = trainOutputList, validation_data = (validInputList, validOutputList), 
                     validation_split = 0, shuffle = True, batch_size = batchSize, epochs = maxEpoch, verbose = 1, 
                     callbacks = createCallbacks(modelFile, earlyStoppingPatience))
############################################################################################################
def loadModel(modelType, modelFile, testInput, taskWeights, noOfFeatures, dropoutRate, layerNum = 4, 
              outputChannelNos = [], outputTypes = []):
    inputHeight = testInput.shape[1]
    inputWidth = testInput.shape[2]
    channelNo = testInput.shape[3]
    
    if modelType == 'unet':
        model = deepModels.unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, layerNum, 
                                noOfFeatures, dropoutRate, taskWeights)
    elif modelType == 'simple-cascade' or modelType == 'input-cascade':
        model = deepModels.cascaded(modelType, inputHeight, inputWidth, channelNo, outputChannelNos[0], outputChannelNos[1],
                                layerNum, noOfFeatures, dropoutRate, taskWeights) 
    model.load_weights(modelFile)
    return model
############################################################################################################
def testModel(model, testInput, weightMapNo):
    testInputList = []
    testInputList.append(testInput)
    for i in range(weightMapNo):
        weightNullMap = np.ones((testInput.shape[0], testInput.shape[1], testInput.shape[2]))
        testInputList.append(weightNullMap)
    
    predictions = model.predict(testInputList, batch_size = 1, verbose = 1)
    return predictions
############################################################################################################
def trainUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPaths, valImageNames, valInputPath, valOutputPaths,
              inputExtension, isGray, outputExtensions, outputTypes, outputChannelNos,
              lossTypes, taskWeights, layerNum, noOfFeatures, dropoutRate, learningRate):
    
    [trInputList, trOutputList] = taskLists4UNet(trImageNames, trInputPath, inputExtension, isGray, trOutputPaths, 
                                                 outputExtensions, outputTypes, lossTypes)
    [valInputList, valOutputList] = taskLists4UNet(valImageNames, valInputPath, inputExtension, isGray, valOutputPaths, 
                                                 outputExtensions, outputTypes, lossTypes)
    
    trainModel(modelType, modelFile, trInputList, trOutputList, valInputList, valOutputList, taskWeights, layerNum, 
               noOfFeatures, dropoutRate, outputChannelNos, outputTypes, learningRate)
############################################################################################################
def testUnet(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPaths, inputExtension, isGray, taskWeights, 
             layerNum, noOfFeatures, dropoutRate, outputTypes, outputChannelNos, outputExtensions):
    [tsInputs, setOutputs, tsNames, outputTypes] = inputOutput.readOneDataset(tsImageNames, tsInputPath, 
                                                                              inputExtension, isGray, tsOutputPaths, 
                                                                              outputExtensions, outputTypes)
    
    model = loadModel(modelType, modelFile, tsInputs, taskWeights, noOfFeatures, dropoutRate, layerNum, 
                      outputChannelNos, outputTypes)
    probs = testModel(model, tsInputs, len(outputTypes))
    return [probs, tsNames]
############################################################################################################
