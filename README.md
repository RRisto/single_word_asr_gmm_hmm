# Implementation of GMM-HMM for speech Recognition using hmmlearn python package

Idea is to generate model which could recognize single words from short speech segments. 
I use GMM HMM for model.

![alt text](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/images/model.png)

[This](https://medium.com/@RistoHinno/single-word-speech-recognition-892c7e01f5fc) is medium article which explaines what and how.

Part of code is from https://github.com/jayaram1125/Single-Word-Speech-Recognition-using-GMM-HMM- 
I've refactored code and added some more features:
 - added MFCC delta and delta-delta features to increase accuracy of the model
 - script to record test audio to test your model(s)
 - trained model on original data from original repository but also took bunch of data from 
 [Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)  
 - just for testing aligned Speech Command Dataset to gain higher accuracy
 
My trained models accuracy information is in [models/accuracies](https://github.com/RRisto/single_word_asr_gmm_hmm/tree/master/models/accuracies)
 directory. Original models are not included as they are too big. Only example fruit names model is in models [directory](https://github.com/RRisto/single_word_asr_gmm_hmm/tree/master/models.
  If you want to use them see example predict_google.py. 
You can record your own voice using record_test_audio.py

## Training Google Speech Commands Dataset model (original)

- Download speech data (like [Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)).
Data should be in folders, each folder should have a name of the label/command/word spoken
in particular directory
- Prepare data for training and testing using [notebook](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/2.0_prep_google_experiment.ipynb) 
This should be similar to original [suggestions](https://arxiv.org/abs/1804.03209) how to make data for training and testing. Note that testing and validation file lists are in [data/]https://github.com/RRisto/single_word_asr_gmm_hmm/tree/master/data folder
- Train model using [train_hmm_google_orig.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/train_hmm_google_orig.py) or other train scripts as a template
- Predict on test data using [predict_google_orig.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/predict_google_orig.py) script
- Test your model using microphone by running script [listen_mic_predict.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/listen_mic_predict.py)

Another [script](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/train_hmm_google.py)  uses data from
 Google Speech Commands Datasets but has only few categories for quicker training (it doesn't have unknown word and noise category) 

## Training very small fruit names dataset

Original data, good for debugging, not very useful for real-life speech recognition.
- unzip data [file](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/data/audio.tar.gz)
- Train model using [train_hmm_fruits.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/train_hmm_fruits.py) or other train scripts as a template
- Test your model using microphone by running script [listen_mic_predict.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/listen_mic_predict.py) as template


## Aligning
This is just experiment I made. Original alignment was very good but this might improve model performance.

If you wan to align data and use it for training:
 - Download 
[Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
 - Run [1.0_prep_data4aligning.ipynb](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/1.0_prep_data4aligning.ipynb)
 - Download/install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)
 - Download LibriSpeech [lexicon](https://montreal-forced-aligner.readthedocs.io/en/latest/) (you can create your own also)
 - Run aligner using following template (in command line): bin/mfa_train_and_align /path/to/dataset_prepared_in_first_step
  /path/to/librispeech/lexicon.txt /path/to/aligned/dataset This part takes few hours (in usual Windows laptop)
 - Run [1.1_generate_aligned_audio_files_risto.ipynb](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/1.1_generate_aligned_audio_files_risto.ipynb) - this will create chunks from original audio which contain only 
 part where command was said
 - Train new model example is in [train_hmm_google_aligned.py](https://github.com/RRisto/single_word_asr_gmm_hmm/blob/master/train_hmm_google_aligned.py)
  
