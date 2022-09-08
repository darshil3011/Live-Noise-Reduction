# Importing project dependencies

import streamlit as st
#import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
#from scipy.io.wavfile import write
import util_functions as ufs
import time
import torchaudio
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio

# Setting config option for deployment

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Think Crisp - AI Noise Removal')
st.subheader('Custom AI engine powered by Think In Bytes')

# UI design

nav_choice = st.sidebar.radio('Navigation', ['Home'], index=0)

_param_dict = {}  # Used for getting plot related information
_path_to_model = 'utils/models/auto_encoders_for_noise_removal_production.h5'  # Path to pre-trained model
target_file = 'utils/outputs/preds.wav'  # target file for storing model.output

model = pretrained.dns48()
#model = dns48-11decc9d8e3f0998.th
#model = 'model'

if nav_choice == 'Home':
  
    st.info('Upload your audio sample below')
    audio_sample = st.file_uploader('Audio Sample', ['wav'])  # Get audio sample as an input from users
            
    if audio_sample:
        
        st.info('Uploaded audio sample')
        st.audio(audio_sample)
    
        
        try:
        
            wav, sr = torchaudio.load(audio_sample)
            wav = convert_audio(wav, sr, model.sample_rate, model.chin)
            st.info('model loaded - AI is analyzing your audio file')
            with torch.no_grad():
                denoised = model(wav[None])[0]
    
            torchaudio.save(target_file, denoised.data.cpu(), model.sample_rate)
            #st.info('model converted file !')
            #st.info(denoised.data.cpu().numpy().shape)  
            st.success('Audio after noise removal')
            st.audio(target_file)
            
         
        except Exception as e:
            print(e, type(e))
        
