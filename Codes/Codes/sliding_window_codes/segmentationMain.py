import os
import numpy as np
import sys, getopt
from keras.utils import np_utils

import taskAwareMain
import OCTMain
############################################################################################################
def main(argv):
    
    serverStr = argv[0]
    channel_name = argv[1]
    foldNo = int(argv[2])
    runNo = int(argv[3])
    learningRate = float(argv[4])
    trainStr = argv[5]
    gpu = argv[6]
    
    if len(argv) == 7:
        gpu = argv[6]
    else:
        gpu = '2'
    
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    OCTMain.OCTSegmentation('unetR', foldNo, runNo, trainStr, learningRate, serverStr,channel_name)
############################################################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
############################################################################################################
