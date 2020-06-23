from pathlib import Path
from sklearn.metrics import classification_report
from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

#testset similar to original in testset
test_files = list(Path('data/audio_google_original/test').rglob('*.wav'))
model = HMMSpeechRecog.unpickle('models/google_orig_hmm.pkl')
predicted_labels = model.predict_files(test_files)
predicted_labels = [label[0].name for label in predicted_labels]
original_labels = [str(fl).split('\\')[-2] for fl in test_files]
report = classification_report(original_labels, predicted_labels)
conf_mat = model.get_confusion_matrix(original_labels, predicted_labels, list(set(original_labels)))

reports = f'{report}\n {conf_mat.to_string()}'
Path('models/accuracies/accuracy_google_orig_hmm_testdata.txt').write_text(reports)
print(f'{reports}')
