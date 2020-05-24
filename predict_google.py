from speech_recognition.HMMSpeechRecog import HMMSpeechRecog

model= HMMSpeechRecog.unpickle('models/google_hmm.pkl')
predicted_labels = model.predict(['data/test_data/output.wav'])
print(f'predicted label {predicted_labels}')