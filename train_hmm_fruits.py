from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

model = HMMSpeechRecog()
model.train(3, 2)
model.test('models/accuracies/accuracy_fruit_hmm.txt')
predicted_labels = model.predict(['data/test_data/apple15.wav'])
print(f'predicted label {predicted_labels}')

model.pickle('models/fruit_hmm.pkl')
model2 = HMMSpeechRecog.unpickle('models/fruit_hmm.pkl')
predicted_labels2 = model2.predict(['data/test_data/apple15.wav'])
print(f'predicted label {predicted_labels2}')

predicted_labels3 = model2.predict(['data/test_data/output.wav'])
print(f'predicted label {predicted_labels3}')

predicted_labels4 = model2.predict(['data/test_data/pineapple_risto3.wav'])
print(f'predicted label {predicted_labels4}')

model2.calc_mean_entropy()
