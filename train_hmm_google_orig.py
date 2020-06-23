from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

# uses data from https://www.tensorflow.org/datasets/catalog/speech_commands
#have smaller validation set as we have separate val set. Mostly based on https://arxiv.org/abs/1804.03209
model = HMMSpeechRecog(filespath='data/audio_google_original/train',val_p=0.05, add_mfcc_delta_delta=False)
model.train(4, 2, m_n_iter=100)
model.test('models/accuracies/accuracy_google_orig_hmm.txt')

model.pickle('models/google_orig_hmm.pkl')
