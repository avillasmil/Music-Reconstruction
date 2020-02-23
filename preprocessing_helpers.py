#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:02:30 2019

@author: alejandro villasmil 
"""

#Import dependencies
import h5py
import pickle
import gzip
from scipy.signal import decimate
from scipy.io import wavfile

#Read:_EEG function that reads in HDF5 files
def read_eeg():
    #Reading in Preprocessed Data
    filename = 'OpenMIIR-Perception-512Hz.hdf5'
    f = h5py.File(filename, 'r')
    
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]
    c_group_key = list(f.keys())[2]
    d_group_key = list(f.keys())[3]
    
    # Get the data/features/subjects/targets
    data = list(f[a_group_key])
    indices = list(f[b_group_key])
    subjects = list(f[c_group_key])
    targets = list(f[d_group_key])
    
    #Unpacking the Metadata
    h = gzip.open('OpenMIIR-Perception-512Hz.hdf5.meta.pklz','rb')
    meta_data = pickle.load(h)
    
    #Returning all variables
    return data,indices,subjects,targets,meta_data

#Read_audio: function that reads in audio file from provided path
def read_audio(audio_path):
    fs, data = wavfile.read(audio_path)
    return fs, data
    
#Downsample audio test data
def downsample(audio, factor):
    down = decimate(audio, factor, ftype='iir', axis=-1, zero_phase=True)
    return down
