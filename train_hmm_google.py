from speech_recognition.HMMSpeechRecog import HMMSpeechRecog
#uses data from https://www.tensorflow.org/datasets/catalog/speech_commands
model = HMMSpeechRecog(filespath='data/audio_google')
model.train(4, 2)
model.test()
predicted_labels = model.predict(['data/audio_google/down/0a7c2a8d_nohash_0.wav'])
print(f'predicted label {predicted_labels}')

model.pickle('models/google_hmm.pkl')
