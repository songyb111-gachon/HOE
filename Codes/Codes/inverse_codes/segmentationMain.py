import os
import numpy as np
import sys, getopt
from keras.utils import np_utils

import taskAwareMain
import OCTMain
############################################################################################################
def main(argv):
    
    serverStr = argv[0]
    modelName = argv[1]
    foldNo = int(argv[2])
    runNo = int(argv[3])
    learningRate = float(argv[4])
    trainStr = argv[5]
    gpu = argv[6]
    
    if len(argv) == 7:
        gpu = argv[6]
    else:
        gpu = '0'
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    if modelName == 'fundus':
        fundusMain.FundusVesselSegmentation('unetR', foldNo, runNo, trainStr, learningRate, serverStr)
    elif modelName == 'fundusG':
        fundusGrayMain.FundusGrayVesselSegmentation('unetR', foldNo, runNo, trainStr, learningRate, serverStr)
    elif modelName == 'SimpleCascade':
        taskAwareMain.TaskAwareGlandSegmentation('simple-cascade', foldNo, runNo, trainStr, learningRate, serverStr)
    elif modelName == 'InputCascade':
        taskAwareMain.TaskAwareGlandSegmentation('input-cascade', foldNo, runNo, trainStr, learningRate, serverStr)
    elif modelName == 'oct':
        OCTMain.OCTSegmentation('unetR', foldNo, runNo, trainStr, learningRate, serverStr, coordinate=0)
############################################################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
############################################################################################################
