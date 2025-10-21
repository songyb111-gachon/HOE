import deepModels
import inputOutput
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

############################################################################################################
def createCallbacks(modelFile, earlyStoppingPatience):
    checkpoint = ModelCheckpoint(modelFile, monitor = 'val_loss', verbose = 1, save_best_only = True, 
                                 save_weights_only = True, mode = 'auto', period = 1)
    earlystopping = EarlyStopping(patience = earlyStoppingPatience, monitor = 'val_loss', verbose = 1, 
                                  restore_best_weights = True)
    return [checkpoint, earlystopping]
############################################################################################################
def plotConvergencePlots(hist, modelFile):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plotFile = modelFile[:-5] + '.png'
    plt.savefig(plotFile)
############################################################################################################
def trainModel(modelType, modelFile, trainInputList, trainOutputList, validInputList, validOutputList, 
               noOfFeatures, dropoutRate, layerNum = 4, earlyStoppingPatience = 50, batchSize = 1, maxEpoch = 1000):
    inputHeight = trainInputList[0].shape[1]
    inputWidth = trainInputList[0].shape[2]
    
    if modelType == 'single':
        model = deepModels.unetSingleRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate)
    elif modelType == 'multi':
        model = deepModels.unetMultiRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate)
    elif modelType == 'cascaded':
        taskWeights = [0.67, 0.67, 0.33, 0.33, 0.33, 0.33, 1.0]
        model = deepModels.cascaded(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate, taskWeights)
    model.summary()

    
    hist = model.fit(x = trainInputList, y = trainOutputList, validation_data = (validInputList, validOutputList), 
                     validation_split = 0, shuffle = True, batch_size = batchSize, epochs = maxEpoch, verbose = 1,
                     callbacks = createCallbacks(modelFile, earlyStoppingPatience))
    plotConvergencePlots(hist, modelFile)
############################################################################################################
def loadModel(modelType, modelFile, testInput, noOfFeatures, dropoutRate, layerNum = 4):
    inputHeight = testInput.shape[1]
    inputWidth = testInput.shape[2]
    
    if modelType == 'single':
        model = deepModels.unetSingleRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate)
    elif modelType == 'multi':
        model = deepModels.unetMultiRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate)
    elif modelType == 'cascaded':
        taskWeights = [0.67, 0.67, 0.33, 0.33, 0.33, 0.33, 1.0]
        model = deepModels.cascaded(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate, taskWeights)
        
    model.load_weights(modelFile)
    return model
############################################################################################################
def testModel(model, testInput, weightMapNo = 1):
    testInputList = []
    testInputList.append(testInput)
    for i in range(weightMapNo):
        weightNullMap = np.ones((testInput.shape[0], testInput.shape[1], testInput.shape[2]))
        testInputList.append(weightNullMap)
    
    predictions = model.predict(testInputList, batch_size = 4, verbose = 1)
    return predictions
############################################################################################################
