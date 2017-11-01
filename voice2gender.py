'''
Description: This program is for gender classification
using voice. It takes in path to a wav file, and then
return back 0/1 based on
0 ==> if voice is of a male person
1 ==> if voice is of a female person

Details: This classifier has been trained on
TIMIT and TIDIGITS database and has been tested
to have an accuracy of around 85%

In terms of implementation details, it uses
a simple MFCC approach with a 4 layer perceptron
model with relu as an activation and sigmoid in
the final layer.
--------------------------------------------
Author: Shahan Ali Memon
Mentors: Rita Singh, Bhiksha Raj
Copyright (c) 2017 Carnegie Mellon University
--------------------------------------------
'''

import numpy as np
import sys
import os
import sox
import preprocess
import scipy.io.wavfile as wav
import numpy as np

from mfcc import *
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from collections import Counter

def compute_mfcc(file_name):
    print file_name
    rate, data = wav.read(file_name);
    mfcc = MFCC(nfilt = 40, ncep = 13, samprate = rate,
                wlen = 0.0256, frate = 100,
                lowerf=133.33334, upperf=6855.4976)
    mfcc = mfcc.sig2s2mfc(data)
    return mfcc

if __name__ == "__main__":
    argv = sys.argv[1:]
    #Accept only 1 argument i.e. the wav file
    if(len(argv) != 1):
        print("Usage: python voice2age.py <full path to wav file>")
        sys.exit()
    else:
        model = load_model("mfcc_ws15_4gender_gender.h5")
        tfm = sox.Transformer()
        #Downsampling the audio to 16000 KHz
        tfm.convert(samplerate=16000)
        tfm.build(argv[0],'downsampled.wav')
        #Now we need to compute mfcc
        mfcc = compute_mfcc('downsampled.wav')
        os.remove('downsampled.wav')
        #Now that we have computed mfcc, we preprocess it
        #We are processing a 15 frame window
        preprocessed_x = preprocess.preprocess(mfcc,15)
        #Now that we have preprocessed x, let us classify it
        #print(Counter(map(lambda pred:1 if pred >= 0.5 else 0,model.predict(preprocessed_x))))
        gender =Counter(map(lambda pred:1 if pred >= 0.5 else 0,
                                     model.predict(preprocessed_x))).most_common()[0][0]
        print gender
        
        
        
        
        
        
        
        
        
