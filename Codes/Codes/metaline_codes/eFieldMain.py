import inputOutput
import deepModels
import os
import numpy as np
import sys, getopt
import tensorflow
import eFieldTasks
############################################################################################################
def callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr):
    networkName = modelType
    if trainStr == 'tr':
        eFieldTasks.trainUnetForEFieldEstimation(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate)
    else:
        if modelType == 'single':
            outputNo = 1
        elif modelType == 'multi':
            outputNo = 6
        elif modelType == 'cascaded':
            outputNo = 7
        [predictions, actual, tsNames, resPath] = eFieldTasks.testUnetForEFieldEstimation(modelType, pathPrefix, networkName, 
                                                                             runNo, layerNum, featNum, dropoutRate, outputNo)
        if modelType == 'single':
            outputNo = 1
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,0], '_pr_x1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,1], '_pr_x2')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,2], '_pr_y1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,3], '_pr_y2')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,4], '_pr_z1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[:,:,:,5], '_pr_z2')
        elif modelType == 'multi' or modelType == 'cascaded':
            inputOutput.saveProbabilities(resPath, tsNames, predictions[0][:,:,:,0], '_pr_x1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[1][:,:,:,0], '_pr_x2')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[2][:,:,:,0], '_pr_y1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[3][:,:,:,0], '_pr_y2')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[4][:,:,:,0], '_pr_z1')
            inputOutput.saveProbabilities(resPath, tsNames, predictions[5][:,:,:,0], '_pr_z2')
            if modelType == 'cascaded':
                labels = inputOutput.findLabels(predictions[6])
                inputOutput.saveSegmentationLabels(resPath, tsNames, labels, '_lb')
        
        return [predictions, actual]
############################################################################################################
def main(argv):
    pathPrefix = '../'
    layerNum = 4
    featNum = 32
    dropoutRate = 0.2
    
    gpu = argv[0]
    runNo = int(argv[1])
    trainStr = argv[2]
    
    modelType = 'cascaded'
    
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
    
    
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr)
############################################################################################################
if __name__ == "__main__":
   main(sys.argv[1:])
############################################################################################################
