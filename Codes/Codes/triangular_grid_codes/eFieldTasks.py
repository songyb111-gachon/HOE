import inputOutput
import trainTestModels
import calculateLoss
import os
import numpy as np
#from keras.utils import np_utils

############################################################################################################
def returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo):
    modelName = networkName + '_L' + str(layerNum) + '_F' + str(featNum) + '_run' + str(runNo)
    modelFile = pathPrefix + 'models/' + modelName + '.hdf5'
    resultFile = pathPrefix + 'results/' + modelName
    
    trNames = inputOutput.listAllPNG(pathPrefix + 'data_normalized/training/')
    valNames = inputOutput.listAllPNG(pathPrefix + 'data_normalized/validation/')
    tsNames = inputOutput.listAllPNG(pathPrefix + 'data_normalized/test/')
    
    return [modelFile, resultFile, trNames, valNames, tsNames]
############################################################################################################
def taskLists4Training(modelType, pathPrefix, trNames, valNames):
    [trInputs, trOutputs, trNames, mean_std_tr] = inputOutput.readOnePillarDataset(pathPrefix + 'data_normalized/training/', trNames, True, False)
    [valInputs, valOutputs, valNames, mean_std_val] = inputOutput.readOnePillarDataset(pathPrefix + 'data_normalized/validation/', valNames, True, False)
    [trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'same')
    
    print(trOutputs.shape)
    print(valOutputs.shape)
    print(trWeights.shape)
    print(valWeights.shape)
    
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
            [trWeights, valWeights] = calculateLoss.trValWeights(trInputs[:,:,:,0], valInputs[:,:,:,0], 'same')
            trOutTmp = inputOutput.createCategoricalOutput(trInputs[:,:,:,0], True, False)
            trInputList.append(trWeights) #trInputList.append(trInputs[:,:,:,0]) 
            trOutputList.append(trOutTmp)
            
            valOutTmp = inputOutput.createCategoricalOutput(valInputs[:,:,:,0], True, False)
            valInputList.append(valWeights) #valInputList.append(valInputs[:,:,:,0])
            valOutputList.append(valOutTmp)
    
    return [trInputList, trOutputList, valInputList, valOutputList]
############################################################################################################
def trainUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, transfer_learn, model_name):
    [modelFile, _, trNames, valNames, _] = returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)
    [trInList, trOutList, valInList, valOutList] = taskLists4Training(modelType, pathPrefix, trNames, valNames)
    hist = trainTestModels.trainModel(modelType, modelFile, trInList, trOutList, valInList, valOutList, featNum, dropoutRate, transfer_learn, model_name, layerNum)
    return hist
############################################################################################################
def testUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, outputNo, transfer_learn, model_name):
    [modelFile, resultFile, _, _, tsNames] = returnPillarTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)    
    [tsInputs, tsOutputs, tsNames, mean_std_test] = inputOutput.readOnePillarDataset(pathPrefix + 'data_normalized/test/', tsNames, True, False)
    model = trainTestModels.loadModel(modelType, modelFile, tsInputs, featNum, dropoutRate, transfer_learn, model_name, layerNum ) 
    predictions = trainTestModels.testModel(model, tsInputs, outputNo)
    return [predictions, tsOutputs, tsNames, resultFile, mean_std_test, tsInputs]
############################################################################################################
def calculate_mse(prediction_matrix, actual):
    print(np.shape(prediction_matrix))
    print(np.shape(actual))
    prediction_matrix = np.reshape(prediction_matrix, (actual.shape[0], actual.shape[1], actual.shape[2], actual.shape[3]))
    tot = (actual.shape[0])*(actual.shape[1]-8)*(actual.shape[2]-8)*(actual.shape[3])
    mse = np.square(actual-prediction_matrix)
    mse_x1 = (np.sum(mse[:,2:62, 2:62,0]))/tot
    mse_x2 = (np.sum(mse[:,2:62, 2:62,1]))/tot
    mse_y1 = (np.sum(mse[:,2:62, 2:62,2]))/tot
    mse_y2 = (np.sum(mse[:,2:62, 2:62,3]))/tot
    mse_z1 = (np.sum(mse[:,2:62, 2:62,4]))/tot
    mse_z2 = (np.sum(mse[:,2:62, 2:62,5]))/tot
    mse_matrix = np.array([mse_x1, mse_x2, mse_y1, mse_y2, mse_z1, mse_z2])
    mse_total = np.sum(mse_matrix)
    mse_matrix = np.append(mse_matrix,mse_total)
    return mse_matrix


def calculate_reverse_path_error(prediction, test_input):
    test_input = np.reshape(test_input,(test_input.shape[0],test_input.shape[1],test_input.shape[2]))
    error_matrix = np.absolute(prediction-test_input)
    error_matrix = np.mean(error_matrix,(1,2))
    error_total = np.mean(error_matrix)
    return [error_matrix, error_total]
    
    
def write_model(date, layerNum, featNum, runNo, network_type, channel_pred, channel_actual, hist, rev_pred=0, rev_actual=0):
    model_name = date + '_L' + str(layerNum) + '_F' + str(featNum) + '_' + network_type + '_run' + str(runNo)
    mse_matrix = calculate_mse(channel_pred, channel_actual)
    if network_type == 'cascaded':
        [error_matrix, error_total] = calculate_reverse_path_error(rev_pred, rev_actual)
    epochs = hist.params['epochs']
    final_loss = hist.history['loss'][-1]    
    final_val_loss = hist.history['val_loss'][-1]
    
    f  = open('Model_Results.txt' , 'a+')
    f.write('Model Name: ')     
    f.write(model_name)
    f.write('\n')
    f.write('Epoch Number: ')     
    f.write(str(epochs))
    f.write('\n')
    f.write('Final Loss: ')     
    f.write(str(final_loss))
    f.write('\n')
    f.write('Final Validation Loss: ')     
    f.write(str(final_val_loss))
    f.write('\n')
    f.write('MSE values for the test set: ')     
    f.write(np.array_str(mse_matrix))
    f.write('\n')
    f.close()
    if network_type == 'cascaded':
        f  = open('Model_Results.txt' , 'a+')
        f.write('Average error values of reverse path for the test set: ')     
        f.write(np.array_str(error_matrix))
        f.write('\n')
        f.write('Total average error for reverse path: ')     
        f.write(str(error_total))
        f.write('\n')
        f.write('\n')
        f.close()
    
     
    
    
    








