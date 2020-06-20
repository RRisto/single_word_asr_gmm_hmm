from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

# uses data from https://www.tensorflow.org/datasets/catalog/speech_commands
model = HMMSpeechRecog(filespath='data/audio_google', add_mfcc_delta_delta=False)
model.train(4, 2, m_n_iter=100)
model.test('models/accuracies/accuracy_google_hmm.txt')
predicted_labels = model.predict_files(['data/audio_google/down/0a7c2a8d_nohash_0.wav'])
print(f'predicted label {predicted_labels}')

model.pickle('models/google_hmm.pkl')
