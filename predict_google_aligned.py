from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

model = HMMSpeechRecog.unpickle('models/google_aligned_hmm.pkl')
predicted_labels = model.predict_files(['data/test_data/output.wav'])

print(f'predicted label {predicted_labels}')

# example if you want to remove original training data to save space
model.features = None
model.pickle('models/google_hmm_aligend_wo_data.pkl')