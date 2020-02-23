#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:59:28 2019

@author: alejandrovillasmil

"""

#Import dependencies
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


#Import helper functions
from training_helpers import *
from preprocessing_helpers import *
from nn import NeuralNetwork

#Main 
if __name__ == "__main__":
   
    ######################################################################
    #                     Importing Audio and EEG Data
    ######################################################################

    # Get current working directory       
    cwd = os.getcwd()
    
    #Reading audio files 
    os.chdir(os.path.join(cwd, 'Audio'))
    
    #This will be the master audio to train on 
    audiofile = 1
    audio_fs, audio = read_audio(os.path.join(cwd, 'Audio', str(audiofile) + ".wav"))
    
    #Reading eeg files
    os.chdir(os.path.join(cwd, 'EEG'))
    eeg_data,indices,subjects,targets,meta_data = read_eeg()
    eeg_fs = 512
    
    #Back to main and ready for preparation
    os.chdir(cwd)
    
    ######################################################################
    #                            Data Preparation
    ######################################################################
    audio = audio[:,0]
    phi = 82500
    time = 6.8710
    audio = audio[phi:int(phi+audio_fs*time)]
    fs = eeg_fs
  
    #This time around we're going to be iterating through all the songs for one patient 
    #instead of all patients for one song and traing i tlike that 
    indices = []
    songs = [] 
    for i in range(len(meta_data)):
        if meta_data[i]['subject'] == 'P01' and meta_data[i]['trial_type']=='perception' :
            indices.append(i)
            songs.append(meta_data[i]['stimulus_id'])
        #%%
    #Dimensionality reduction
    k = 1
    bumble = KernelPCA(n_components = k, kernel='linear') 
    
    ######################################################################
    #                            NN Training
    ######################################################################
    test_audio = 24
    nets = []
    print('Total Trials = around ' + str(len(indices)))
    #Iterating through each patient who has listed to that one song:
    for i, trial in enumerate(indices):
        #Read in audio:
        audio_fs, audio = read_audio(os.path.join(cwd, 'Audio', str(songs[i]) + ".wav"))
        audio = audio[:,0]
        phi = 82500
        time = 6.8710
        audio = audio[phi:int(phi+audio_fs*time)]
        
        #Select EEG Sample
        sample = eeg_data[trial]
        
        #Downsampling audio to get to 3518 samples as well
        L = len(audio)
        pad = L%(len(sample[0]))
        pad_audio = np.concatenate((audio[:,None], np.zeros((pad,1))), axis = 0)
        factor = np.floor(len(pad_audio)/len(sample[0]))
        audio_down = downsample(pad_audio[:,0], int(factor))
        audio_down = audio_down[:len(sample[0])]
        
        #Neural Network Training
        print('Started Trial: ' + str(i))
        #If test data, ski
        if songs[i] == test_audio:
            continue
        
        if i == 0: #Create Net
            Y = audio_down
            X_dim = 1
            Y_dim = 1
            layers = [X_dim, 50, 50, 50, 50, 50, 50, 50, Y_dim]
            
            #Reduce EEG Dimension
            X = bumble.fit_transform(sample[:,:,0].T)
            
            #Create Net
            m = NeuralNetwork(X, Y[:,None], layers)  
            
            #Train first time
            m.train(X, Y[:,None], nIter = 1000)
            
        else: 
            #Reduce EEG Dimension
            X = bumble.fit_transform(sample[:,:,0].T)
            Y = audio_down
            
            #Train first time
            m.train(X, Y[:,None], nIter = 1000)
            
            #Once trained, append current net to a list so we can use it for future predictions 
            nets.append(m)
        print('Finished Trial: ' + str(i))
            
    #%%
    ######################################################################
    #                        Testing with Neural Net
    ######################################################################
    
    #Importing Test Audio
    os.chdir(os.path.join(cwd, 'Audio'))
    audiofile = 24
    audio_fs, audio = read_audio(os.path.join(cwd, 'Audio', str(audiofile) + ".wav"))
    #Back to main and ready for preparation
    audio = audio[:,0]
    phi = 130500
    time = 6.8710
    audio = audio[phi:int(phi+audio_fs*time)]
    test_eeg = eeg_data[-1]
    
    from scipy import signal
    X = bumble.fit_transform(test_eeg[:,:,0].T)
    pred = nets[-1].predict(X)
    audio_reconstruct_up = signal.resample(pred, int(len(pred)*factor))
    audio_reconstruct_up = audio_reconstruct_up * np.max(audio)/ np.max(audio_reconstruct_up)
    
    
    #Reconstructing the Audio Fil
    #Interpolating so that it upsamples
    from scipy import signal
    #Renormalizing so that it matches audio input: 

    
    #%%
    #Writing out file:
    from scipy.io.wavfile import write
    write('ninety_song_r' + str(audiofile) +' .wav', 44100, audio_reconstruct_up)
    write('ninety_song_t' + str(audiofile) + '.wav', 44100, audio)
    
    #%% Plotting Predictions vs Actual Data

    #Audio Reconstructed
    plt.figure()
    plt.plot(audio_reconstruct_up)
    plt.title("Reconstructed Audio for NN Depth: " + str(len(layers)))
    plt.savefig("RA_Train_Depth" + str(len(layers)) + ".png")
    
    #Audio Original 
    plt.figure()
    plt.plot(audio)
    plt.title("Original Audio Waveform")
    plt.savefig("OA_train_" + str(audiofile) + ".png")
    





