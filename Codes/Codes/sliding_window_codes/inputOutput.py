import numpy as np
from keras.utils import np_utils
import os
from os import listdir
import errno
import cv2

############################################################################################################
def normalizeImage(img):
    normImg = np.zeros(img.shape)
    for i in range(img.shape[2]):
        if(img[:, :, i].std() != 0):
            normImg[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / (img[:, :, i].std())
    return normImg
############################################################################################################
def readOneGoldMap(goldFileName, imageExtension, outputType):
    if imageExtension == 'png':
        gold = cv2.imread(goldFileName)
        gold = gold[:, :, 0]
    else:
        file = open(goldFileName, 'r')
        gold = file.read()
        if (outputType == 'BC' or outputType == 'MC'):
            gold = [list(map(int, line.split(' ')[0:-1])) for line in gold.split('\n')[0:-1]]
        else:
            gold = [list(map(float, line.split(' ')[0:-1])) for line in gold.split('\n')[0:-1]]
        file.close()
        
    gold = np.asarray(gold)
    return gold
############################################################################################################
def readOneImage(imageName, inputTypes, inputPath, inputExtension, isGray, outputPath = '', 
                 outputExtensions = '', outputTypes = ''):
    
    img = cv2.imread(inputPath + imageName + inputTypes[0])
    tmp = img[:, :, 0]

    inputLen = len(inputTypes)
    allImg = np.zeros((tmp.shape[0], tmp.shape[1], inputLen))
    allImg[:, :, 0] = tmp
    for i in range(1, inputLen):
        img = cv2.imread(inputPath + imageName + inputTypes[i] + '.png')
        tmp = img[:, :, 0]
        allImg[:, :, i] = tmp
    allImg = normalizeImage(allImg)
    
    
    outputLen = len(outputTypes)
    allOut = np.zeros((tmp.shape[0], tmp.shape[1], outputLen))
    
    if outputPath != '':

        for i in range(1, outputLen):
            img = np.loadtxt(outputPath[1] + imageName + ".fdmap" + str(i))
            out = (img - img.mean()) / (img.std())
            allOut[:, :, i-1] = out

        img = cv2.imread(outputPath[0] + imageName + '.png')
        out = (img[:, :, 0] > 0).astype(np.int_)
        out = np.asarray(out)
        allOut[:, :, outputLen-1] = out
    
    return [allImg, allOut]
############################################################################################################
def readOneDataset(imageNames, inputPath, inputExtension, isGray, outputPaths = '', outputExtensions = '', outputTypes = ''):
    d_names = []
    d_inputs = []
    d_outputs = []
    
    inputTypes = ['.png']
    
    outputTypes = ['fdmap1', 'fdmap2', 'hfl']

    
    for fname in imageNames:
        
        [img, gold] = readOneImage(fname, inputTypes, inputPath, inputExtension, isGray, outputPaths, 
                                   outputExtensions, outputTypes)
        d_outputs.append(gold)
        d_inputs.append(img)
        d_names.append(fname)
    
    d_inputs = np.asarray(d_inputs)
    d_outputs = np.asarray(d_outputs)
    return [d_inputs, d_outputs, d_names, outputTypes]
    
############################################################################################################
def createCategoricalOutput(gold, outputType):
    if outputType == 'BC':
        gold = gold > 0
    
    categoricalOutput = np_utils.to_categorical(gold)
    return categoricalOutput
############################################################################################################
def listAllImageFiles(imageNames, imageDirPath, imageExtension):
    fileList = listdir(imageDirPath)
    tmpStr = '.' + imageExtension
    for i in range(len(fileList)):
        if fileList[i][-4::] == tmpStr:
            imageNames.append(fileList[i][:-4])
    return imageNames
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
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
############################################################################################################
def saveAsDouble(dirPath, names, values, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(values)):
        fname = dirPath + names[i] + postfix
        np.savetxt(fname, values[i], fmt = '%1.4f')
    print('Values are saved')
############################################################################################################
def saveAsInteger(dirPath, names, values, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(values)):
        fname = dirPath + names[i] + postfix
        np.savetxt(fname, values[i], fmt = '%1.0f')
    print('Values are saved')
############################################################################################################
def generateOutputNames(savePathPrefix, modelType, taskName, layerNo, firstFeatureNo, foldNo, runNo):
    networkName = modelType + '_' + taskName
    currName =  networkName + '_L' + str(layerNo) + '_F' + str(firstFeatureNo) + '_fold' + str(foldNo) + '_run' + str(runNo)
    modelFile = savePathPrefix + 'models/' + currName + '.hdf5'
    resultDirectory = savePathPrefix + 'results/' + currName + '/'
    return [modelFile, resultDirectory]
############################################################################################################
