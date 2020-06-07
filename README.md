# Implementation of GMM-HMM for speech Recognition using hmmlearn python package.
Forked from https://github.com/jayaram1125/Single-Word-Speech-Recognition-using-GMM-HMM- 
I've refactored code and added some more features:
 - added MFCC delta features to increase accuracy of the model
 - script to record test audio to test your model(s)
 - trained model on original data from original repository but also took bunch of data from 
 [Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)  
 - Just for testing aligned Speech Command Dataset to gain higher accuracy
 
Trained models are in models/ directory, If you want to use them see example predict_google.py. 
You can record your own voice using record_test_audio.py

## Training

- Download speech data (like [Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)).
Data should be in folders, each folder should have a name of the label/command/word said on files in particual directory
- Train model using train_hmm_google.py or other train scripts as a template
- Record test audio using record_test_audio.py 
- Test your model on recorded audio using predict_google.py as template

## Aligning

If you wan to align data and use it for training:
 - Download 
[Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
 - Run 1.0_prep_data4aligning.ipynb
 - Download/install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)
 - Download LibriSpeech [lexicon](https://montreal-forced-aligner.readthedocs.io/en/latest/) (you can create your own also)
 - Run aligner using following template (in command line): bin/mfa_train_and_align /path/to/dataset_prepared_in_first_step
  /path/to/librispeech/lexicon.txt /path/to/aligned/dataset This part takes few hours (in usual Windows laptop)
 - Run 1.1_generate_aligned_audio_files_risto.ipynb - this will create chunks from original audio which contain only 
 part where command was said
 - Train new model example is in train_hmm_google_aligned.py
  
