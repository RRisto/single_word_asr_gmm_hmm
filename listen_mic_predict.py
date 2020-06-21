from speech_recogn.listen_mic_predict import listen_predict_mic

# MODEL_PATH = 'models/google_aligned_hmm.pkl'
MODEL_PATH = 'models/google_hmm.pkl'

listen_predict_mic(MODEL_PATH, phrase_time_limit=1.5)
