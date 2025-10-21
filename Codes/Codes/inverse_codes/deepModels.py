import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Concatenate
from keras.layers.core import Layer, Dense, Activation, Flatten, Reshape, Permute, Lambda
from tensorflow.keras import optimizers, losses

import calculateLoss

############################################################################################################
def unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same',  
                                kernel_initializer = 'he_uniform')(blockInput)  # glorot_uniform
    blockOutput = Dropout(dropoutRate)(blockOutput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'he_uniform')(blockOutput)
    return blockOutput
############################################################################################################
def unetOneEncoderBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    conv = unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return [conv, pool]
############################################################################################################
def unetOneDecoderBlock(blockInput, longSkipInput, noOfFeatures, filterSize, dropoutRate):
    upR = concatenate([UpSampling2D(size = (2, 2))(blockInput), longSkipInput], axis = 3)
    conv = unetOneBlock(upR, noOfFeatures, filterSize, dropoutRate)
    return conv
############################################################################################################
def oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate):
    if layerNum >= 1:
        [conv1, pool1] = unetOneEncoderBlock(inputs, noOfFeatures[0], filterSize, dropoutRate)
    if layerNum >= 2:
        [conv2, pool2] = unetOneEncoderBlock(pool1,  noOfFeatures[1], filterSize, dropoutRate)
    if layerNum >= 3:
        [conv3, pool3] = unetOneEncoderBlock(pool2,  noOfFeatures[2], filterSize, dropoutRate)
    if layerNum >= 4:
        [conv4, pool4] = unetOneEncoderBlock(pool3,  noOfFeatures[3], filterSize, dropoutRate)
    if layerNum >= 5:
        [conv5, pool5] = unetOneEncoderBlock(pool4,  noOfFeatures[4], filterSize, dropoutRate)
    if layerNum >= 6:
        [conv6, pool6] = unetOneEncoderBlock(pool5,  noOfFeatures[5], filterSize, dropoutRate)
    if layerNum >= 7:
        [conv7, pool7] = unetOneEncoderBlock(pool6,  noOfFeatures[6], filterSize, dropoutRate)

    if layerNum == 1:
        return [conv1, pool1]
    elif layerNum == 2:
        return [conv1, conv2, pool2]
    elif layerNum == 3:
        return [conv1, conv2, conv3, pool3]
    elif layerNum == 4:
        return [conv1, conv2, conv3, conv4, pool4]
    elif layerNum == 5:
        return [conv1, conv2, conv3, conv4, conv5, pool5]
    elif layerNum == 6:
        return [conv1, conv2, conv3, conv4, conv5, conv6, pool6]
    elif layerNum == 7:
        return [conv1, conv2, conv3, conv4, conv5, conv6, conv7, pool7]
############################################################################################################
def oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, 
                   conv1, conv2=[], conv3=[], conv4=[], conv5=[], conv6=[], conv7=[]):
    if layerNum == 7:
        deconv7 = unetOneDecoderBlock(lastConv, conv7, noOfFeatures[6], filterSize, dropoutRate)
    elif layerNum == 6:
        deconv6 = unetOneDecoderBlock(lastConv, conv6, noOfFeatures[5], filterSize, dropoutRate)
    elif layerNum == 5:
        deconv5 = unetOneDecoderBlock(lastConv, conv5, noOfFeatures[4], filterSize, dropoutRate)
    elif layerNum == 4:
        deconv4 = unetOneDecoderBlock(lastConv, conv4, noOfFeatures[3], filterSize, dropoutRate)
    elif layerNum == 3:
        deconv3 = unetOneDecoderBlock(lastConv, conv3, noOfFeatures[2], filterSize, dropoutRate)
    elif layerNum == 2:
        deconv2 = unetOneDecoderBlock(lastConv, conv2, noOfFeatures[1], filterSize, dropoutRate)
    elif layerNum == 1:
        deconv1 = unetOneDecoderBlock(lastConv, conv1, noOfFeatures[0], filterSize, dropoutRate)

    if layerNum > 6:
        deconv6 = unetOneDecoderBlock(deconv7, conv6, noOfFeatures[5], filterSize, dropoutRate)
    if layerNum > 5:
        deconv5 = unetOneDecoderBlock(deconv6, conv5, noOfFeatures[4], filterSize, dropoutRate)
    if layerNum > 4:
        deconv4 = unetOneDecoderBlock(deconv5, conv4, noOfFeatures[3], filterSize, dropoutRate)
    if layerNum > 3:
        deconv3 = unetOneDecoderBlock(deconv4, conv3, noOfFeatures[2], filterSize, dropoutRate)
    if layerNum > 2:
        deconv2 = unetOneDecoderBlock(deconv3, conv2, noOfFeatures[1], filterSize, dropoutRate)
    if layerNum > 1:
        deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures[0], filterSize, dropoutRate)
    
    return deconv1
############################################################################################################
def outputLayer(lastDeconv, outputWeights, outputType, outputChannelNo, name):
    if outputType == 'BC':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_binary_crossentropy(outputWeights)
    elif outputType == 'MC':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'softmax', name = name)(lastDeconv)
        outputLoss = "categorical_crossentropy"
    else:
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'linear', name = name)(lastDeconv)
        outputLoss = "mean_squared_error"
    return [outputs, outputLoss]
############################################################################################################
def createInputOutputLostLists(inputHeight, inputWidth, inputs, lastDeconvTask1, lastDeconvTask2, lastDeconvTask3,
                               lastDeconvTask4, lastDeconvTask5, outputChannelNos, outputTypes):
    inputList = []
    outputList = []
    lossList = []
    
    inputList.append(inputs)
    
    outputWeights1 = Input(shape = (inputHeight, inputWidth))
    [output1, loss1] = outputLayer(lastDeconvTask1, outputWeights1, outputTypes[0], outputChannelNos[0], 'out1')
    inputList.append(outputWeights1)
    outputList.append(output1)
    lossList.append(loss1)
    
    taskNo = len(outputChannelNos)
    if taskNo >= 2:
        outputWeights2 = Input(shape = (inputHeight, inputWidth))
        [output2, loss2] = outputLayer(lastDeconvTask2, outputWeights2, outputTypes[1], outputChannelNos[1], 'out2')
        inputList.append(outputWeights2)
        outputList.append(output2)
        lossList.append(loss2)
    if taskNo >= 3:
        outputWeights3 = Input(shape = (inputHeight, inputWidth))
        [output3, loss3] = outputLayer(lastDeconvTask3, outputWeights3, outputTypes[2], outputChannelNos[2], 'out3')
        inputList.append(outputWeights3)
        outputList.append(output3)
        lossList.append(loss3)
    if taskNo >= 4:
        outputWeights4 = Input(shape = (inputHeight, inputWidth))
        [output4, loss4] = outputLayer(lastDeconvTask4, outputWeights4, outputTypes[3], outputChannelNos[3], 'out4')
        inputList.append(outputWeights4)
        outputList.append(output4)
        lossList.append(loss4)
    if taskNo >= 5:
        outputWeights5 = Input(shape = (inputHeight, inputWidth))
        [output5, loss5] = outputLayer(lastDeconvTask5, outputWeights5, outputTypes[4], outputChannelNos[4], 'out5')
        inputList.append(outputWeights5)
        outputList.append(output5)
        lossList.append(loss5)
    
    if taskNo == 10:
        lossList[0] = calculateLoss.categorical_crossentropy(outputList[1], outputList[2])

    return [inputList, outputList, lossList]
############################################################################################################
def unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, layerNum, noOfFeatures, 
         dropoutRate, taskWeights, lr = 0.001):
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta(lr = lr)

    inputs = Input(shape = (inputHeight, inputWidth, channelNo), name = 'input')
    
    if layerNum == 1:
        [conv1, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[1], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1)
    elif layerNum == 2:
        [conv1, conv2, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[2], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2)
    elif layerNum == 3:
        [conv1, conv2, conv3, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[3], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3)
    elif layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[4], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4)
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[5], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4,
                                         conv5)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5)
    elif layerNum == 6:
        [conv1, conv2, conv3, conv4, conv5, conv6, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, 
                                                                           dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[6], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6)
    elif layerNum == 7:
        [conv1, conv2, conv3, conv4, conv5, conv6, conv7, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, 
                                                                                  dropoutRate)
        lastConv = unetOneBlock(poolR, noOfFeatures[7], filterSize, dropoutRate)
        lastDeconvTask1 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6, conv7)
        lastDeconvTask2 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6, conv7)
        lastDeconvTask3 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6, conv7)
        lastDeconvTask4 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6, conv7)
        lastDeconvTask5 = oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4, 
                                         conv5, conv6, conv7)

    [inputList, outputList, lossList] = createInputOutputLostLists(inputHeight, inputWidth, inputs, lastDeconvTask1, 
                                                                   lastDeconvTask2, lastDeconvTask3, lastDeconvTask4, 
                                                                   lastDeconvTask5, outputChannelNos, outputTypes)   
    model = Model(inputs = inputList, outputs = outputList)
    model.compile(loss = lossList, loss_weights = taskWeights, optimizer = optimizer, experimental_run_tf_function=False,
                  metrics = ['mean_squared_error'])
    return model
############################################################################################################
def unet4OneMultiClassification(inputs, outputNo, layerNum, noOfFeat, fsize, drate, outName):
    if layerNum == 1:
        [conv1, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[1], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1)
    elif layerNum == 2:
        [conv1, conv2, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[2], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2)
    elif layerNum == 3:
        [conv1, conv2, conv3, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[3], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2, conv3)
    elif layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[4], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2, conv3, conv4)
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[5], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2, conv3, conv4, conv5)
    elif layerNum == 6:
        [conv1, conv2, conv3, conv4, conv5, conv6, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[6], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2, conv3, conv4, conv5, conv6)
    elif layerNum == 7:
        [conv1, conv2, conv3, conv4, conv5, conv6, conv7, poolR] = oneEncoderPath(inputs, layerNum, noOfFeat, fsize, drate)
        lastConv = unetOneBlock(poolR, noOfFeat[7], fsize, drate)
        lastDeconv = oneDecoderPath(lastConv, layerNum, noOfFeat, fsize, drate, conv1, conv2, conv3, conv4, conv5, conv6, conv7)


    if outputNo == 2:
        netOutput = Convolution2D(outputNo, (1, 1), activation = 'softmax', name = outName)(lastDeconv)

    elif outputNo == 1:
        netOutput = Convolution2D(outputNo, (1, 1), activation = 'linear', name = outName)(lastDeconv)

    return netOutput

############################################################################################################    
def cascaded(cascadedType, inputHeight, inputWidth, channelNo, interOutputNo, finalOutputNo, layerNum, noOfFeatures, 
             dropoutRate, taskWeights, lr = 0.001):
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta(learning_rate = lr)
    
    inputs = Input(shape = (inputHeight, inputWidth, channelNo), name = 'input')
    [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath(inputs, layerNum, noOfFeatures, filterSize, dropoutRate)
    lastConv = unetOneBlock(poolR, noOfFeatures[4], filterSize, dropoutRate)
    
    fd_channel = 2
    
    lastDeconv = []
    for i in range(fd_channel):
        lastDeconv.append(oneDecoderPath(lastConv, layerNum, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4))
        
    interOutput = []
    for i in range(fd_channel):
        interOutput.append(Convolution2D(interOutputNo, (1, 1), activation = 'linear', name = "interO" + str(i+1))(lastDeconv[i]))
    
    interWeights = []
    for i in range(fd_channel):
        interWeights.append(Input(shape = (inputHeight, inputWidth), name = 'inter-weights' + str(i+1)))
    
    interLoss = []
    for i in range(fd_channel):
        interLoss.append("mean_squared_error")
    

    if cascadedType == 'simple-cascade':
        inputs2 = concatenate(interOutput, axis = 3)
        
    else:
        interOutput.insert(0,inputs)
        inputs2 = concatenate(interOutput, axis = 3)
    finalOutput = unet4OneMultiClassification(inputs2, finalOutputNo, 
                                              layerNum, noOfFeatures, filterSize, dropoutRate, 'finalO')
    finalWeights = Input(shape = (inputHeight, inputWidth), name = 'final-weights')
    finalLoss = "categorical_crossentropy"
    interWeights.insert(0,inputs)
    interWeights.append(finalWeights)
    interOutput.append(finalOutput)
    
    model = Model(inputs = interWeights, outputs = interOutput[1:])
    
    for i in range(fd_channel-1):
        taskWeights.insert(0, taskWeights[0])
     
    interLoss.append(finalLoss)
    model.compile(loss = interLoss, loss_weights = taskWeights, experimental_run_tf_function=False,
                  optimizer = optimizer, metrics = ['mean_squared_error']) # ['categorical_accuracy']
    return model
