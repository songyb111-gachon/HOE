import numpy as np
from keras.utils import np_utils
import os
from os import listdir
import errno
import cv2
import scipy
from scipy import io

import inputOutput
############################################################################################################
def normalizeImage(img):
    normImg = np.zeros(img.shape)
    if(img.std() != 0):
        normImg = (img - img.mean()) / (img.std())
    return normImg
############################################################################################################
def listAllOCTFiles(imageDirPath, imagePostfix):
    fileList = listdir(imageDirPath)
    postLen = len(imagePostfix)
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-postLen::] == imagePostfix:
            imageNames.append(fileList[i][:-postLen])
    return imageNames
############################################################################################################
def readOneOCT(imageName, inputPath, inputTypes, outputPath, outputTypes): 
    img = cv2.imread(inputPath + imageName + inputTypes[0])
    tmp = img[:, :, 0]
    tmp = (tmp > 0).astype(np.int_)

    inputLen = len(inputTypes)
    allImg = np.zeros((tmp.shape[0], tmp.shape[1], inputLen))
    allImg[:, :, 0] = tmp
    
    allImg = inputOutput.normalizeImage(allImg)
    
    outputLen = len(outputTypes)
    allOut = np.zeros((tmp.shape[0], tmp.shape[1], outputLen))
    
    if outputPath != '':
    
        img = np.loadtxt(outputPath[0] + "random_" + imageName[7:] + outputTypes[0])
        allOut[:, :, 0] = img
        
        for i in range(1, outputLen):
            img = np.loadtxt(outputPath[1] + imageName + ".fdmap" + str(i))
            out = (img - img.mean()) / (img.std())
            allOut[:, :, i] = out
            
    return [allImg, allOut]

############################################################################################################
def readOneOCTDataset(imageNames, inputPath, outputPath):
    
    inputTypes = ['.png']

    outputTypes = ['_x2.txt']

    d_names = []
    d_inputs = []
    d_outputs = []
    
    for fname in imageNames:
        [img, gold] = readOneOCT(fname, inputPath, inputTypes, outputPath, outputTypes)
        d_outputs.append(gold)
        d_inputs.append(img)
        d_names.append(fname)
        
    d_inputs = np.asarray(d_inputs)
    d_outputs = np.asarray(d_outputs)
    return [d_inputs, d_outputs, d_names, outputTypes]
############################################################################################################
