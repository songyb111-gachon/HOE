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
def normalizeWholeSet(array):
    if(array.std() != 0):
        array = (array - array.mean()) / (array.std())
    return array
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

    img = np.loadtxt(inputPath + imageName + inputTypes[0])
    tmp = img

    inputLen = len(inputTypes)
    allImg = np.zeros((tmp.shape[0], tmp.shape[1], inputLen))
    allImg[:, :, 0] = tmp
    
    for i in range(1, inputLen):
       img = np.loadtxt(inputPath + imageName + inputTypes[i])
       allImg[:, :, i] = img
    
    outputLen = len(outputTypes)
    allOut = np.zeros((tmp.shape[0], tmp.shape[1], outputLen))
    
    if outputPath != '':
        
        img = cv2.imread(outputPath + 'pillar_' + imageName[7:] + outputTypes[0])[:,:,0]
        img = (img > 0).astype(np.int_)
        allOut[:, :, 0] = img 
        
        for i in range(1, outputLen):
            img = np.loadtxt(outputPath[1] + imageName + ".fdmap" + str(i))
            out = (img - img.mean()) / (img.std())
            allOut[:, :, i] = out
            
    return [allImg, allOut]

############################################################################################################
def readOneOCTDataset(imageNames, inputPath, outputPath, coordinate):
    
    inputTypes = ['_x1.txt', '_x2.txt', '_y1.txt', '_y2.txt', '_z1.txt', '_z2.txt']
    
    outputTypes = ['.png']

    d_names = []
    d_inputs = []
    d_outputs = []
    
    for fname in imageNames:
        [img, gold] = readOneOCT(fname, inputPath, inputTypes, outputPath, outputTypes)
        d_outputs.append(gold)
        d_inputs.append(img)
        d_names.append(fname)
        
    d_inputs = np.asarray(d_inputs)

    for i in range(len(inputTypes)):
      d_inputs[:,:,:,i] = normalizeWholeSet(d_inputs[:,:,:,i])

    d_outputs = np.asarray(d_outputs)
    return [d_inputs, d_outputs, d_names, outputTypes]
############################################################################################################
