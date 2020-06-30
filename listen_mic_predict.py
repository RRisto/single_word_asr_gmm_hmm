from speech_recogn.listen_mic_predict import listen_predict_mic

#turn on and talk to mic
MODEL_PATH = 'models/google_orig_hmm.pkl'

listen_predict_mic(MODEL_PATH, phrase_time_limit=2)
