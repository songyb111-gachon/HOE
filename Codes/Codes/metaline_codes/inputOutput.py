import numpy as np
import os
from os import listdir
import errno
import cv2
import tensorflow

############################################################################################################
def listAllPNG(datasetPath):
    fileList = listdir(datasetPath + 'inputs/')
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-4::] == '.png':
            imageNames.append(fileList[i][:-4])
    return imageNames
############################################################################################################
def normalizeImage(img):
    normImg = np.zeros(img.shape)
    for i in range(img.shape[2]):
        if(img[:, :, i].std() != 0):
            normImg[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / (img[:, :, i].std())
            
    return normImg

############################################################################################################
def normalizematrix(img):
    normImg = np.zeros(img.shape)
    
    normImg[:, :] = (img[:, :] - img[:, :].mean()) / (img[:, :].std())
            
    return normImg

############################################################################################################
def findOutputDimension(gold, binaryClass, ignoreBackground):
    if binaryClass:
        maxNo = 2
    else:
        maxNo = gold.max() + 1
        if ignoreBackground:
            maxNo -= 1
    return maxNo
############################################################################################################
def createCategoricalOutput(gold, binaryClass, ignoreBackground):
    if ignoreBackground:
        gold -= 1
        gold[gold == -1] = 0
        
    if binaryClass:
        gold = gold > 0

    categoricalOutput = tensorflow.keras.utils.to_categorical(gold)
    return categoricalOutput
############################################################################################################
def findLabels(probs):
    imageNo = probs.shape[0]
    inputHeight = probs.shape[1]
    inputWidth = probs.shape[2]
    
    labels = np.zeros((imageNo, inputHeight, inputWidth))
    for i in range(len(probs)):
        labels[i] = np.argmax(probs[i], axis = 2)
    return labels
############################################################################################################
def readOneGoldMap(goldFileName, headerFlag, integerFlag):
    file = open(goldFileName, 'r')
    gold = file.read()
    if (integerFlag):
        gold = [list(map(int, line.split(',')[0:-1])) for line in gold.split('\n')[headerFlag:-1]]
    else:
        gold = [list(map(float, line.split(','))) for line in gold.split('\n')[headerFlag:-1]]
    gold = np.asarray(gold)
    file.close() 
    
    return gold
############################################################################################################
def readOnePillarImage(datasetPath, imageName, goldFlag):
    imageFileName = datasetPath + 'inputs/' + imageName + '.png'   
    outFileName = datasetPath + 'outputs/' + imageName + '_'
    finalOutputNo = 6

    tmp = cv2.imread(imageFileName)
    tmp = normalizeImage(tmp)
    img = np.zeros((tmp.shape[0], tmp.shape[1], 1))
    img[:, :, 0] = tmp[:, :, 0] 


    if (goldFlag):

        out1 = readOneGoldMap(outFileName + 'x1.txt', 0, False)
        out2 = readOneGoldMap(outFileName + 'x2.txt', 0, False)
        out3 = readOneGoldMap(outFileName + 'y1.txt', 0, False)
        out4 = readOneGoldMap(outFileName + 'y2.txt', 0, False)
        out5 = readOneGoldMap(outFileName + 'z1.txt', 0, False)
        out6 = readOneGoldMap(outFileName + 'z2.txt', 0, False)
        

        
        
        
        

        gold = np.zeros((img.shape[0], img.shape[1], finalOutputNo))
        gold[:, :, 0] = out1        
        gold[:, :, 1] = out2
        gold[:, :, 2] = out3        
        gold[:, :, 3] = out4
        gold[:, :, 4] = out5        
        gold[:, :, 5] = out6
        return [img, gold]
    else:
        return img
############################################################################################################
def readOnePillarDataset(datasetPath, imageNames, goldFlag):
    d_names = []
    d_inputs = []
    d_outputs = []
    
    for fname in imageNames:
        if (goldFlag):
            [img, gold] = readOnePillarImage(datasetPath, fname, True)
            d_outputs.append(gold)
        else:
            img = readOnePillarImage(datasetPath, fname, False)
        d_inputs.append(img)
        d_names.append(fname)
        
    d_inputs = np.asarray(d_inputs)
    if (goldFlag):
        d_outputs = np.asarray(d_outputs)
        return [d_inputs, d_outputs, d_names]
    else:
        return [d_inputs, d_names]
############################################################################################################
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
############################################################################################################
def saveProbabilities(dirPath, names, probs, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(probs)):
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, probs[i], fmt = '%1.4f')
    print('Probabilities are saved')
############################################################################################################
def saveSegmentationLabels(dirPath, names, labels, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(labels)):
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, labels[i], fmt = '%1.0f')
    print('Labels are saved')
############################################################################################################
