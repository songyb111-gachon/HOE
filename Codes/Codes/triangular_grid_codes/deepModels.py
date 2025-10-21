import tensorflow as tf
from keras import backend as K
import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, ZeroPadding2D
from tensorflow.keras.layers import Layer, Dense, Activation, Flatten, Reshape, Permute, Lambda
from tensorflow.keras import  optimizers, losses

import calculateLoss

############################################################################################################
def unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'glorot_uniform')(blockInput)
    blockOutput = Dropout(dropoutRate)(blockOutput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'glorot_uniform')(blockOutput)
    return blockOutput
############################################################################################################
def unetOneEncoderBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    conv = unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return [conv, pool]
############################################################################################################
def unetOneBlock_padding(blockInput, noOfFeatures, filterSize, dropoutRate):
    #blockOutput = ZeroPadding2D(padding=2, data_format="channels_last")(blockInput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same' ,kernel_initializer = 'glorot_uniform')(blockInput)
    blockOutput = Dropout(dropoutRate)(blockOutput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'glorot_uniform')(blockOutput)
    return blockOutput
############################################################################################################
def unetOneEncoderBlock_padding(blockInput, noOfFeatures, filterSize, dropoutRate):
    blockOutput = ZeroPadding2D(padding=4, data_format="channels_last")(blockInput)
    conv = unetOneBlock_padding(blockOutput, noOfFeatures, filterSize, dropoutRate)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return [conv, pool]
############################################################################################################
def oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate):
    [conv1, poolR] = unetOneEncoderBlock(inputs, noOfFeatures, filterSize, dropoutRate)
    [conv2, poolR] = unetOneEncoderBlock(poolR, 2 * noOfFeatures, filterSize, dropoutRate)
    [conv3, poolR] = unetOneEncoderBlock(poolR, 4 * noOfFeatures, filterSize, dropoutRate)
    [conv4, poolR] = unetOneEncoderBlock(poolR, 8 * noOfFeatures, filterSize, dropoutRate)
    return [conv1, conv2, conv3, conv4, poolR]
############################################################################################################
def oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate):
    [conv1, poolR] = unetOneEncoderBlock(inputs, noOfFeatures, filterSize, dropoutRate)
    [conv2, poolR] = unetOneEncoderBlock(poolR, 2 * noOfFeatures, filterSize, dropoutRate)
    [conv3, poolR] = unetOneEncoderBlock(poolR, 4 * noOfFeatures, filterSize, dropoutRate)
    [conv4, poolR] = unetOneEncoderBlock(poolR, 8 * noOfFeatures, filterSize, dropoutRate)
    [conv5, poolR] = unetOneEncoderBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
    return [conv1, conv2, conv3, conv4, conv5, poolR]
############################################################################################################
def unetOneDecoderBlock(blockInput, longSkipInput, noOfFeatures, filterSize, dropoutRate):
    upR = concatenate([UpSampling2D(size = (2, 2))(blockInput), longSkipInput], axis = 3)
    conv = unetOneBlock(upR, noOfFeatures, filterSize, dropoutRate)    
    return conv
############################################################################################################
def oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate):
    deconv4 = unetOneDecoderBlock(lastConv, conv4, 8 * noOfFeatures, filterSize, dropoutRate)
    deconv3 = unetOneDecoderBlock(deconv4, conv3, 4 * noOfFeatures, filterSize, dropoutRate)
    deconv2 = unetOneDecoderBlock(deconv3, conv2, 2 * noOfFeatures, filterSize, dropoutRate)
    deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures, filterSize, dropoutRate)
    return deconv1
############################################################################################################
def oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate):
    deconv5 = unetOneDecoderBlock(lastConv, conv5, 16 * noOfFeatures, filterSize, dropoutRate)
    deconv4 = unetOneDecoderBlock(deconv5, conv4, 8 * noOfFeatures, filterSize, dropoutRate)
    deconv3 = unetOneDecoderBlock(deconv4, conv3, 4 * noOfFeatures, filterSize, dropoutRate)
    deconv2 = unetOneDecoderBlock(deconv3, conv2, 2 * noOfFeatures, filterSize, dropoutRate)
    deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures, filterSize, dropoutRate)
    return deconv1
############################################################################################################
def outputLayer(lastDeconv, outputWeights, outputType, outputChannelNo, name):
    if outputType == 'C':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'softmax', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_categorical_crossentropy(outputWeights)
    else:
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'linear', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_mse(outputWeights)
    return [outputs, outputLoss]
############################################################################################################
def createInputOutputLostListsForSixTasks(inputHeight, inputWidth, inputs, lastD1, lastD2, lastD3, lastD4, lastD5, lastD6):
    out1 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out1')(lastD1)
    out2 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out2')(lastD2)
    out3 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out3')(lastD3)
    out4 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out4')(lastD4)
    out5 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out5')(lastD5)
    out6 = Convolution2D(1, (1, 1), activation = 'linear', name = 'out6')(lastD6)
    
    outW1 = Input(shape = (inputHeight, inputWidth), name = 'outW_1')
    outW2 = Input(shape = (inputHeight, inputWidth), name = 'outW_2')
    outW3 = Input(shape = (inputHeight, inputWidth), name = 'outW_3')
    outW4 = Input(shape = (inputHeight, inputWidth), name = 'outW_4')
    outW5 = Input(shape = (inputHeight, inputWidth), name = 'outW_5')
    outW6 = Input(shape = (inputHeight, inputWidth), name = 'outW_6')
    
    outL1 = calculateLoss.weighted_mse(outW1)
    outL2 = calculateLoss.weighted_mse(outW2)
    outL3 = calculateLoss.weighted_mse(outW3)
    outL4 = calculateLoss.weighted_mse(outW4)
    outL5 = calculateLoss.weighted_mse(outW5)
    outL6 = calculateLoss.weighted_mse(outW6)
    
    inputList = [inputs, outW1, outW2, outW3, outW4, outW5, outW6]
    outputList = [out1, out2, out3, out4, out5, out6]
    lossList = [outL1, outL2, outL3, outL4, outL5, outL6]
    
    return [inputList, outputList, lossList]
############################################################################################################
def unetSingleRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate):
    outputChannelNo = 6
    inputChannelNo = 1

    filterSize = (3, 3)
    optimizer = optimizers.Adadelta()
    inputs = Input(shape = (inputHeight, inputWidth, inputChannelNo), name = 'input')
    outputWeights = Input(shape = (inputHeight, inputWidth), name = 'out_weight')
    if layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 32 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        
    outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'linear', name = 'output')(lastDeconv)
    outputLoss = calculateLoss.weighted_mse(outputWeights)

    model = Model(inputs = [inputs, outputWeights], outputs = [outputs])
    model.compile(loss = [outputLoss], optimizer = optimizer, metrics = ['mean_squared_error'])
    return model
############################################################################################################
def unetMultiRegression(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate, transfer_learn, model_name):
    outputChannelNo = 6
    inputChannelNo = 1
    
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta()
    inputs = Input(shape = (inputHeight, inputWidth, inputChannelNo), name = 'input')
    outputWeights = Input(shape = (inputHeight, inputWidth), name = 'out_weight')
    if layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv5 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv6 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 32 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv5 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv6 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)

    [inputList, outputList, lossList] = createInputOutputLostListsForSixTasks(inputHeight, inputWidth, inputs, lastDeconv1, 
                                                            lastDeconv2, lastDeconv3, lastDeconv4, lastDeconv5, lastDeconv6)
    model = Model(inputs = inputList, outputs = outputList)
    model.compile(loss = lossList, optimizer = optimizer, metrics = ['mean_squared_error'], experimental_run_tf_function=False)
    return model
############################################################################################################
def cascaded(inputHeight, inputWidth, layerNum, noOfFeatures, dropoutRate, taskWeights, transfer_learn, model_name):
    outputChannelNo = 6
    inputChannelNo = 1
    
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta()
    inputs = Input(shape = (inputHeight, inputWidth, inputChannelNo), name = 'input')
    outputWeights = Input(shape = (inputHeight, inputWidth), name = 'out_weight')

    if layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv5 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv6 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 32 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv5 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv6 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)

    [inputList, outputList, lossList] = createInputOutputLostListsForSixTasks(inputHeight, inputWidth, inputs, lastDeconv1, lastDeconv2, lastDeconv3, lastDeconv4, lastDeconv5, lastDeconv6)        
    reconsInput = concatenate(outputList, axis = 3)
    if layerNum == 4:
        [conv_r1, conv_r2, conv_r3, conv_r4, pool_r] = oneEncoderPath4(reconsInput, noOfFeatures, filterSize, dropoutRate)
        lastConv_r = unetOneBlock(pool_r, 16 * noOfFeatures, filterSize, dropoutRate)
        reconsDeconv = oneDecoderPath4(lastConv_r, conv_r4, conv_r3, conv_r2, conv_r1, noOfFeatures, filterSize, dropoutRate)
    elif layerNum == 5:
        [conv_r1, conv_r2, conv_r3, conv_r4, conv_r5, pool_r] = oneEncoderPath5(reconsInput, noOfFeatures, 
                                                                                filterSize, dropoutRate)
        lastConv_r = unetOneBlock(pool_r, 32 * noOfFeatures, filterSize, dropoutRate)
        reconsDeconv = oneDecoderPath5(lastConv_r, conv_r5, conv_r4, conv_r3, conv_r2, conv_r1, 
                                                                   noOfFeatures, filterSize, dropoutRate)
        
    reconsOutput = Convolution2D(2, (1, 1), activation = 'softmax', name = 'out_recons')(reconsDeconv)
    reconsWeight = Input(shape = (inputHeight, inputWidth), name = 'out_weight')
    reconsLoss = calculateLoss.weighted_categorical_crossentropy(reconsWeight)

    inputList.append(reconsWeight)
    outputList.append(reconsOutput)
    lossList.append(reconsLoss)
    
    model = Model(inputs = inputList, outputs = outputList)
    if transfer_learn==1:
        model.load_weights(model_name)
    model.compile(loss = lossList, loss_weights = taskWeights, optimizer = optimizer, metrics = ['mean_squared_error'], experimental_run_tf_function=False)
    
    return model
############################################################################################################

