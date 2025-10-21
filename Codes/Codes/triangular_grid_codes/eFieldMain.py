import inputOutput
import os
import numpy as np
import sys, getopt
import tensorflow
import eFieldTasks
from tensorflow.python.framework.ops import disable_eager_execution


############################################################################################################
def callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, transfer_learn, model_name, date):
    networkName = modelType
    if trainStr == 'tr':
        hist = eFieldTasks.trainUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, transfer_learn, model_name)
    
    if modelType == 'single':
        outputNo = 1
    elif modelType == 'multi':
        outputNo = 6
    elif modelType == 'cascaded':
        outputNo = 7
    [predictions, actual, tsNames, resPath, mean_std_test, tsInputs] = eFieldTasks.testUnetForEFieldEstimation(modelType, pathPrefix, networkName, 
                                                                         runNo, layerNum, featNum, dropoutRate, outputNo, transfer_learn, model_name )
    if modelType == 'single':
        outputNo = 1
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,0], '_pr_x1')
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,1], '_pr_x2')
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,2], '_pr_y1')
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,3], '_pr_y2')
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,4], '_pr_z1')
        inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,5], '_pr_z2')
    elif modelType == 'multi' or modelType == 'cascaded':
        pr_x1=inputOutput.saveProbabilities(resPath, tsNames, predictions[0][:,:,:,0], '_pr_x1',0, mean_std_test)
        pr_x2=inputOutput.saveProbabilities(resPath, tsNames, predictions[1][:,:,:,0], '_pr_x2',1, mean_std_test)
        pr_y1=inputOutput.saveProbabilities(resPath, tsNames, predictions[2][:,:,:,0], '_pr_y1',2, mean_std_test)
        pr_y2=inputOutput.saveProbabilities(resPath, tsNames, predictions[3][:,:,:,0], '_pr_y2',3, mean_std_test)
        pr_z1=inputOutput.saveProbabilities(resPath, tsNames, predictions[4][:,:,:,0], '_pr_z1',4, mean_std_test)
        pr_z2=inputOutput.saveProbabilities(resPath, tsNames, predictions[5][:,:,:,0], '_pr_z2',5, mean_std_test)
        prediction_matrix=np.array([pr_x1, pr_x2, pr_y1, pr_y2, pr_z1, pr_z2])
        if modelType == 'cascaded':
            labels = inputOutput.findLabels(predictions[6])
            inputOutput.saveSegmentationLabels(resPath, tsNames, labels, '_lb')
            eFieldTasks.write_model(date, layerNum, featNum, runNo, modelType, prediction_matrix, actual, hist, labels, tsInputs)
        elif modelType == 'multi':
            eFieldTasks.write_model(date, layerNum, featNum, runNo, modelType, prediction_matrix, actual, hist)
            
    return [predictions, actual]
############################################################################################################
def main(argv):
    pathPrefix = '../'
    layerNum = 5
    featNum = 32
    dropoutRate = 0.2
    gpu = argv[0]
    runNo = int(argv[1])
    trainStr = argv[2]
    transfer_learn=0
    model_name = pathPrefix + 'models/' +  'cascaded' + '_L' + str(5) + '_F' + str(32) + '_run' + str(27) + '.hdf5'
    date = '1_Feb(downsampled_data)'
    disable_eager_execution()
    
    modelType = 'cascaded'
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, transfer_learn, model_name, date)
############################################################################################################
if __name__ == "__main__":
   main(sys.argv[1:])
############################################################################################################
