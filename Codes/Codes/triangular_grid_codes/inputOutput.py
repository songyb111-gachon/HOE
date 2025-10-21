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
    mean=img[:, :].mean()
    std=img[:, :].std()
    return [normImg, mean, std]

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
        gold = [list(map(int, line.split('\t')[0:-1])) for line in gold.split('\n')[headerFlag:-1]]
    else:
        gold = [list(map(float, line.split('\t'))) for line in gold.split('\n')[headerFlag:-1]]
    gold = np.asarray(gold)
    file.close() 
    
    return gold
############################################################################################################
def readOnePillarImage(datasetPath, imageName, goldFlag, test):
    imageFileName = datasetPath + 'inputs/' + imageName + '.png'   
    if test == True:
        outFileName = datasetPath + 'outputs/' + 'test'+imageName[4:] + '_'
    else:
        outFileName = datasetPath + 'outputs/' + 'pillars'+imageName[7:] + '_'

    finalOutputNo = 6

    tmp = cv2.imread(imageFileName)
    

    tmp = normalizeImage(tmp)
    img = np.zeros((tmp.shape[0]+4, tmp.shape[1]+4, 1))
    img[2:62, 2:62, 0] = tmp[:,:, 0]

    if (goldFlag):

        [out1, mean_x1, std_x1] = normalizematrix(readOneGoldMap(outFileName + 'x1.txt', 0, False))
        [out2, mean_x2, std_x2] = normalizematrix(readOneGoldMap(outFileName + 'x2.txt', 0, False))
        [out3, mean_y1, std_y1] = normalizematrix(readOneGoldMap(outFileName + 'y1.txt', 0, False))
        [out4, mean_y2, std_y2] = normalizematrix(readOneGoldMap(outFileName + 'y2.txt', 0, False))
        [out5, mean_z1, std_z1] = normalizematrix(readOneGoldMap(outFileName + 'z1.txt', 0, False))
        [out6, mean_z2, std_z2] = normalizematrix(readOneGoldMap(outFileName + 'z2.txt', 0, False))
        
        mean_std_matrix=np.array([[mean_x1, mean_x2,mean_y1, mean_y2, mean_z1, mean_z2],[std_x1, std_x2, std_y1, std_y2, std_z1, std_z2]])

        gold = np.zeros((img.shape[0], img.shape[1], finalOutputNo))
        gold[2:62, 2:62, 0] = out1        
        gold[2:62, 2:62, 1] = out2
        gold[2:62, 2:62, 2] = out3        
        gold[2:62, 2:62, 3] = out4
        gold[2:62, 2:62, 4] = out5        
        gold[2:62, 2:62, 5] = out6



        return [img, gold, mean_std_matrix]
    else:
        return img
############################################################################################################
def readOnePillarDataset(datasetPath, imageNames, goldFlag, test):
    d_names = []
    d_inputs = []
    d_outputs = []
    count=0
    mean_std=np.zeros((len(imageNames),2,6))
    for fname in imageNames:
        if (goldFlag):
            [img, gold, mean_std_matrix] = readOnePillarImage(datasetPath, fname, True, test)
            d_outputs.append(gold)
            mean_std[count,:,:]=mean_std_matrix
            count= count+1
        else:
            [img, gold, mean_std_matrix] = readOnePillarImage(datasetPath, fname, False, test)
            mean_std[count,:,:]=mean_std_matrix
            count= count+1
        d_inputs.append(img)
        d_names.append(fname)
        
    d_inputs = np.asarray(d_inputs)
    if (goldFlag):
        d_outputs = np.asarray(d_outputs)
        return [d_inputs, d_outputs, d_names, mean_std]
    else:
        return [d_inputs, d_names, mean_std]
############################################################################################################
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
############################################################################################################
def saveProbabilities(dirPath, names, probs, postfix, task_info, mean_std):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    for i in range(len(probs)):
        probs[i]=(probs[i]*mean_std[i,1,task_info])+mean_std[i,0,task_info]
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, probs[i] , fmt = '%1.4f')
    print('Probabilities are saved')
    return probs
############################################################################################################
def saveSegmentationLabels(dirPath, names, labels, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(labels)):
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, labels[i], fmt = '%1.0f')
    print('Labels are saved')
############################################################################################################
