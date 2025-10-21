import inputOutput
import trainTestModels
import calculateLoss
import os
import numpy as np

############################################################################################################
def returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo):
    modelName = networkName + '_L' + str(layerNum) + '_F' + str(featNum) + '_run' + str(runNo)
    modelFile = pathPrefix + 'models/' + modelName + '.hdf5'
    resultFile = pathPrefix + 'results/' + modelName
    
    trNames = inputOutput.listAllPNG(pathPrefix + 'data_downsampled/training/')
    valNames = inputOutput.listAllPNG(pathPrefix + 'data_downsampled/validation/')
    tsNames = inputOutput.listAllPNG(pathPrefix + 'data_downsampled/test/')
    
    return [modelFile, resultFile, trNames, valNames, tsNames]
############################################################################################################
def taskLists4Training(modelType, pathPrefix, trNames, valNames):
    [trInputs, trOutputs, trNames] = inputOutput.readOnePillarDataset(pathPrefix + 'data_downsampled/training/', trNames, True)
    [valInputs, valOutputs, valNames] = inputOutput.readOnePillarDataset(pathPrefix + 'data_downsampled/validation/', valNames, True)
    [trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'same')
    
    if modelType == 'single':
        trInputList = [trInputs, trWeights]
        valInputList = [valInputs, valWeights]
        trOutputList = [trOutputs]
        valOutputList = [valOutputs]
    if modelType == 'multi' or modelType == 'cascaded':
        outputNo = 6
        trInputList = [trInputs]
        valInputList = [valInputs]
        trOutputList = []
        valOutputList = []
        for i in range(outputNo):
            trInputList.append(trWeights)
            trOutTmp = np.zeros((trOutputs.shape[0], trOutputs.shape[1], trOutputs.shape[2], 1))
            trOutTmp[:, :, :, 0] = trOutputs[:, :, :, i]
            trOutputList.append(trOutTmp)
            
            valInputList.append(valWeights)
            valOutTmp = np.zeros((valOutputs.shape[0], valOutputs.shape[1], valOutputs.shape[2], 1))
            valOutTmp[:, :, :, 0] = valOutputs[:, :, :, i]
            valOutputList.append(valOutTmp)
        if modelType == 'cascaded':
            [trWeights, valWeights] = calculateLoss.trValWeights(trInputs[:,:,:,0], valInputs[:,:,:,0], 'class-weighted')
            trOutTmp = inputOutput.createCategoricalOutput(trInputs[:,:,:,0], True, False)
            trInputList.append(trWeights)
            trOutputList.append(trOutTmp)
            
            valOutTmp = inputOutput.createCategoricalOutput(valInputs[:,:,:,0], True, False)
            valInputList.append(valWeights)
            valOutputList.append(valOutTmp)
    
    return [trInputList, trOutputList, valInputList, valOutputList]
############################################################################################################
def trainUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate):
    [modelFile, _, trNames, valNames, _] = returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)
    [trInList, trOutList, valInList, valOutList] = taskLists4Training(modelType, pathPrefix, trNames, valNames)
    trainTestModels.trainModel(modelType, modelFile, trInList, trOutList, valInList, valOutList, featNum, dropoutRate, layerNum)
############################################################################################################
def testUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, outputNo):
    [modelFile, resultFile, _, _, tsNames] = returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)    
    [tsInputs, tsOutputs, tsNames] = inputOutput.readOnePillarDataset(pathPrefix + 'data_downsampled/test/', tsNames, True)
    model = trainTestModels.loadModel(modelType, modelFile, tsInputs, featNum, dropoutRate, layerNum) 
    predictions = trainTestModels.testModel(model, tsInputs, outputNo)
    return [predictions, tsOutputs, tsNames, resultFile]
############################################################################################################
