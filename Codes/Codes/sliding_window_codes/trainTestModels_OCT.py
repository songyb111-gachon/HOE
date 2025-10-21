import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
import cv2

import inputOutput
import inputOutput_OCT
import deepModels
import calculateLoss
import trainTestModels
import mat73
import scipy

############################################################################################################
def returnAuxInfo(taskNo):
    outputTypes = ['R']
    taskWeights = [1.0]
    outputChannelNos = [1]    
    
    if taskNo >= 2:
        for i in range(1, taskNo):
            outputTypes.append('R')
            taskWeights.append(1.0)
            outputChannelNos.append(1)
            
    return [outputChannelNos, outputTypes, taskWeights]
############################################################################################################
def taskLists4UNet_tr(imageNames, inputPath, outputPath, channel_name):
    setInputList = []
    setOutputList = []
    
    channel_dict = {'x1':'1','x2':'2','y1':'3','y2':'4','z1':'5','z2':'6'}
    train_file_str=inputPath+'dataset_'+channel_dict[channel_name]+'.mat'
    setOutputs=mat73.loadmat(train_file_str)['field']
    
    im_height, im_width, taskNo = setOutputs.shape[:3]
    tSize =setOutputs.shape[3]
    
    setInput=[]
    setOutput=[]
    for i in range(setOutputs.shape[3]):
        setOutput.append(setOutputs[:,:,:,i])
    setOutput = np.array(setOutput)
    
    setInputs=mat73.loadmat(inputPath+'index.mat')['index']
    for i in range(setInputs.shape[2]):
        setInput.append(setInputs[:,:,i])
    setInput=np.array(setInput)
    setInput = setInput.reshape(tSize, im_height ,im_width, 1)
    setInput = (setInput-np.mean(setInput))/np.std(setInput) ##normalize input
        
    setInputList.append(setInput)
    for i in range(taskNo):
       
        currW = calculateLoss.calculateLossWeightsForOneDataset(setOutput[:, :, :, i], 'same')
        currOut = setOutput[:, :, :, i].reshape(tSize, im_height ,im_width, 1)
            
        setInputList.append(currW)
        setOutputList.append(currOut)
        
    return [setInputList, setOutputList, taskNo]
    ############################################################################################################
def taskLists4UNet_val(imageNames, inputPath, outputPath, channel_name):
    setInputList = []
    setOutputList = []
    
    channel_dict = {'x1':'1','x2':'2','y1':'3','y2':'4','z1':'5','z2':'6'}
    val_file_str=inputPath+'dataset_valid_'+channel_dict[channel_name]+'.mat'
    setOutputs=mat73.loadmat(val_file_str)['field']
    
    im_height, im_width, taskNo = setOutputs.shape[:3]
    tSize =setOutputs.shape[3]
    
    setInput=[]
    setOutput=[]
    for i in range(setOutputs.shape[3]):
        setOutput.append(setOutputs[:,:,:,i])
    setOutput = np.array(setOutput)
      
    setInputs=mat73.loadmat(inputPath+'index_valid.mat')['index']
    for i in range(setInputs.shape[2]):
        setInput.append(setInputs[:,:,i])
    setInput=np.array(setInput)
    setInput = setInput.reshape(tSize, im_height ,im_width, 1)
    
    im_height, im_width, taskNo = setOutputs.shape[:3]
    tSize =setOutputs.shape[3]
        
    setInputList.append(setInput)
    for i in range(taskNo): 
       
        currW = calculateLoss.calculateLossWeightsForOneDataset(setOutput[:, :, :, i], 'same')
        currOut = setOutput[:, :, :, i].reshape(tSize, im_height ,im_width, 1)
            
        setInputList.append(currW)
        setOutputList.append(currOut)
        
    return [setInputList, setOutputList, taskNo]
############################################################################################################
def trainUnet(modelType, modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath,
              layerNum, noOfFeatures, dropoutRate, learningRate, channel_name):
    [trInputList, trOutputList, taskNo] = taskLists4UNet_tr(trImageNames, trInputPath, trOutputPath, channel_name)
    [valInputList, valOutputList, taskNo] = taskLists4UNet_val(valImageNames, valInputPath, valOutputPath, channel_name)
    [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
    
    trainTestModels.trainModel(modelType, modelFile, trInputList, trOutputList, valInputList, valOutputList, taskWeights, 
                               layerNum, noOfFeatures, dropoutRate, outputChannelNos, outputTypes, learningRate)
############################################################################################################
def testUnet(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate):
    [tsInputs, tsOutputs, tsNames, outputTypes] = inputOutput_OCT.readOneOCTDataset(tsImageNames, tsInputPath, tsOutputPath)
    taskNo = tsOutputs.shape[3]
    [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
    
    model = trainTestModels.loadModel(modelType, modelFile, tsInputs, taskWeights, noOfFeatures, dropoutRate, layerNum,
                                      outputChannelNos, outputTypes)
    probs = trainTestModels.testModel(model, tsInputs, len(outputTypes))
    return [probs, tsNames, taskNo]
############################################################################################################
def testUnet_with_mat_file(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, channel_name, tsResultPath, runNo):
    img_str = 'index_test' + '.mat'
    img_directory = '../results_matlab/' + img_str
    tsInputs = mat73.loadmat(img_directory)
    os.mkdir(tsResultPath)
    str_list=[]
    test_num=5
    for i in range(1,test_num+1):
        index_str = 'index_test' + str(i)   
        mat = np.array(tsInputs[index_str])
        test_index_map=np.zeros([mat.shape[2],mat.shape[0],mat.shape[1],1])
        for j in range(mat.shape[2]):
            test_index_map[j,:,:,0]=mat[:,:,j]
    
        taskNo = 7

        [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
        
        model = trainTestModels.loadModel(modelType, modelFile, test_index_map, taskWeights, noOfFeatures, dropoutRate, layerNum,
                                          outputChannelNos, outputTypes)
        probs = trainTestModels.testModel(model, test_index_map, len(outputTypes))
        
        
        
        save_str = tsResultPath + 'run' + str(runNo) + '_' + str(channel_name) + '_result_test_' + str(i) + '.mat'
        save_str_pred= tsResultPath + 'run' + str(runNo) + '_' + str(channel_name) + '_recombined.mat'
        str_list.append(save_str)
        scipy.io.savemat(save_str, {'field':probs})

    pred_matrix=recombine_Efield(test_num,str_list, taskNo)
    scipy.io.savemat(save_str_pred, {'prediction':pred_matrix})
        
############################################################################################################
def testUnet_with_mat_file_lens(modelType, modelFile, tsImageNames, tsInputPath, tsOutputPath, layerNum, noOfFeatures, dropoutRate, channel_name, tsResultPath, runNo):
    img_directory = '../data/test_bilge_cir/inputs/'
    img_list = os.listdir(img_directory)
    tmp_list = []
    for i in range(1,42):
        for j in range(1,42):
            img = 'circ_' + str(i) + '_' + str(j) + '.png'
            tmp = cv2.imread(img_directory + img, cv2.IMREAD_GRAYSCALE)
            tmp = tmp/255
            tmp_list.append(tmp)
    img_arr = np.asarray(tmp_list)
    mat = np.zeros((img_arr.shape[0],img_arr.shape[1],img_arr.shape[2],1))
    tsResultPath = tsResultPath[:-1] + '_bilge_cir/' 
    os.mkdir(tsResultPath)
    str_list=[]
    test_num=1
    
    mat[:,:,:,0] = img_arr
    
    taskNo = 7

    [outputChannelNos, outputTypes, taskWeights] = returnAuxInfo(taskNo)
        
    model = trainTestModels.loadModel(modelType, modelFile, mat, taskWeights, noOfFeatures, dropoutRate, layerNum,
                                          outputChannelNos, outputTypes)
    probs = trainTestModels.testModel(model, mat, len(outputTypes))
    
    save_str = tsResultPath + 'run' + str(runNo) + '_' + str(channel_name) + '_bilge_circ' + '.mat'
    save_str_pred= tsResultPath + 'run' + str(runNo) + '_' + str(channel_name) + '_bilge_circ' + '_recombined.mat'
    str_list.append(save_str)
    scipy.io.savemat(save_str, {'field':probs})

    pred_matrix=recombine_Efield_lens(test_num,str_list, taskNo)
    scipy.io.savemat(save_str_pred, {'prediction':pred_matrix})
        
############################################################################################################
def recombine_Efield(test_number,str_list, freq_num):
    pred_matrix=np.zeros([2048,2048,7,test_number])
    for test_num in range(test_number):
        mat_file_name=str_list[test_num]
        field = scipy.io.loadmat(mat_file_name)['field']
    
        shape=2400-224
        count=np.ones([128,128])
        num=30
        step_size=64
        offset=64
        image_size=128
        for freq in range(freq_num):
          E_pred=np.zeros([shape,shape])
          
          count_map=np.zeros([shape,shape])
          for i in range(num+1):
              for j in range(num+1):
                p = field[freq,(j)+(num+1)*i,:,:]
                p = np.reshape(p,[256,256]);
                E_pred[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]=E_pred[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]+p[64:192,64:192]

                count_map[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]=count_map[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]+count
          
          count_map=np.where(count_map==0, 1, count_map)   
          E_pred=np.divide(E_pred,count_map)
          E_pred=np.where(E_pred==np.inf, 0, E_pred) 
          E_pred=E_pred[offset:shape-offset,offset:shape-offset]
          pred_matrix[:,:,freq,test_num] = E_pred
          
    return pred_matrix
############################################################################################################
def recombine_Efield_lens(test_number,str_list, freq_num):
    pred_matrix=np.zeros([1409,1409,7,test_number])
    for test_num in range(test_number):
        mat_file_name=str_list[test_num]
        field = scipy.io.loadmat(mat_file_name)['field']
    
        shape=1537
        count=np.ones([128,128])
        num=40
        step_size=32
        offset=64
        image_size=128
        for freq in range(freq_num):
            E_pred=np.zeros([shape,shape])
            count_map=np.zeros([shape,shape])
            
            for i in range(num+1):
                for j in range(num+1):
                    p = field[freq,(j)+(num+1)*i,:,:]
                    p = np.reshape(p,[256,256]);
                    E_pred[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]=E_pred[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]+p[64:192,64:192]
    
                    count_map[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]=count_map[step_size*i+offset:step_size*i+offset+image_size,step_size*j+offset:step_size*j+offset+image_size]+count
            
            count_map=np.where(count_map==0, 1, count_map)   
            E_pred=np.divide(E_pred,count_map)
            E_pred=np.where(E_pred==np.inf, 0, E_pred) 
            E_pred=E_pred[offset:shape-offset,offset:shape-offset]
            pred_matrix[:,:,freq,test_num] = E_pred
          
    return pred_matrix




