import collections
import pickle
import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy.io import wavfile
from operator import itemgetter
from pathlib import Path

from hmmlearn import hmm
from python_speech_features import mfcc, delta
from sklearn.metrics import classification_report, confusion_matrix


class HMMSpeechRecog(object):
    def __init__(self, filespath=Path('data/audio'), val_p=0.2, num_cep=12):
        self.filespath = Path(filespath)
        self.val_p = val_p
        self.num_cep = num_cep
        self._get_filelist_labels()
        self.features = self._get_features()
        self._get_val_index_end()
        self._get_gmmhmmindex_dict()

    def _get_filelist_labels(self):
        self.fpaths = list(self.filespath.rglob('*.wav'))
        self.labels = [file.parent.stem for file in self.fpaths]
        self.spoken = list(set(self.labels))

    def _get_features(self, fpaths=None, eval=False, num_delta=5):
        features = []
        if fpaths is None:
            fpaths = self.fpaths
        for n, file in enumerate(fpaths):
            if n % 10 == 0:
                print(f'working on file nr {n}: {file}')
            samplerate, d = wavfile.read(file)
            mfcc_features = mfcc(d, samplerate=samplerate, numcep=self.num_cep)
            delta_features = delta(mfcc_features, num_delta)
            features.append(np.append(mfcc_features, delta_features, 1))
        if eval:
            return features

        c = list(zip(features, self.labels))
        np.random.shuffle(c)
        features, self.labels = zip(*c)

        print(f'nr of features {len(features)}')
        print(f'nr of labels {len(self.labels)}')
        return features

    def _get_val_index_end(self):
        self.val_i_end = int(len(self.features) * self.val_p)

    def _get_gmmhmmindex_dict(self):
        self.gmmhmmindexdict = {}
        self.indexgmmhmmdict = {}
        index = 0
        for word in self.spoken:
            self.gmmhmmindexdict[word] = index
            self.indexgmmhmmdict[index] = word
            index = index + 1

    def getTransmatPrior(self, inumstates, ibakisLevel):
        transmatPrior = (1 / float(ibakisLevel)) * np.eye(inumstates)

        for i in range(inumstates - (ibakisLevel - 1)):
            for j in range(ibakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. / ibakisLevel

        for i in range(inumstates - ibakisLevel + 1, inumstates):
            for j in range(inumstates - i - j):
                transmatPrior[i, i + j] = 1. / (inumstates - i)

        return transmatPrior

    def initByBakis(self, inumstates, ibakisLevel):
        startprobPrior = np.zeros(inumstates)
        startprobPrior[0: ibakisLevel - 1] = 1 / float((ibakisLevel - 1))
        transmatPrior = self.getTransmatPrior(inumstates, ibakisLevel)
        return startprobPrior, transmatPrior

    def init_model(self, m_num_of_HMMStates, m_bakisLevel):
        self.m_num_of_HMMStates = m_num_of_HMMStates
        self.m_bakisLevel = m_bakisLevel
        self.m_startprobPrior, self.m_transmatPrior = self.initByBakis(self.m_num_of_HMMStates, self.m_bakisLevel)

    def train(self, m_num_of_HMMStates=3, m_bakisLevel=2, m_num_of_mixtures=2, m_covarianceType='diag', m_n_iter=10):
        self.m_num_of_mixtures = m_num_of_mixtures
        self.m_covarianceType = m_covarianceType
        self.m_n_iter = m_n_iter

        self.init_model(m_num_of_HMMStates, m_bakisLevel)
        self.speechmodels = [None] * len(self.spoken)

        for key in self.gmmhmmindexdict:
            self.speechmodels[self.gmmhmmindexdict[key]] = SpeechModel(self.gmmhmmindexdict[key], key,
                                                                       self.m_num_of_HMMStates,
                                                                       self.m_num_of_mixtures, self.m_transmatPrior,
                                                                       self.m_startprobPrior, self.m_covarianceType,
                                                                       self.m_n_iter,
                                                                       self.features[0].shape[1])

        for i in range(self.val_i_end, len(self.features[self.val_i_end:])):
            for j in range(0, len(self.speechmodels)):
                if int(self.speechmodels[j].Class) == int(self.gmmhmmindexdict[self.labels[i]]):
                    self.speechmodels[j].traindata = np.concatenate(
                        (self.speechmodels[j].traindata, self.features[i]))

        for speechmodel in self.speechmodels:
            speechmodel.model.fit(speechmodel.traindata)
        n_spoken = len(self.spoken)
        print(f'Training completed -- {n_spoken} GMM-HMM models are built for {n_spoken} different types of words')

    def get_confusion_matrix(self, real_y, pred_y, labels):
        conf_mat = confusion_matrix(real_y, pred_y, labels=labels)
        df_conf_mat = pd.DataFrame(conf_mat)
        df_conf_mat.columns = labels
        df_conf_mat.index = labels
        return df_conf_mat

    def get_accuracy(self, save_path=None):
        self.accuracy = 0.0
        count = 0
        predicted_labels = []

        print("")
        print("Prediction for test set:")

        for i in range(0, len(self.labels[:self.val_i_end])):
            predicted_label_i = self.m_PredictionlabelList[i]
            predicted_labels.append(self.indexgmmhmmdict[predicted_label_i])
            if self.gmmhmmindexdict[self.labels[i]] == predicted_label_i:
                count = count + 1

        report = classification_report(self.labels[:self.val_i_end], predicted_labels)
        df_conf_mat = self.get_confusion_matrix(self.labels[:self.val_i_end], predicted_labels,
                                                labels=list(set(list(self.labels[:self.val_i_end]) + predicted_labels)))
        print(report)
        print(df_conf_mat.to_string())
        if save_path is not None:
            Path(save_path).write_text(f'nr of files in test set: {count}\n{report}'
                                       f'\nConfusion matrix (y-axis real label, x-axis predicted label):\n'
                                       f'{df_conf_mat.to_string()}')

    def test(self, save_path=None):
        # Testing
        self.m_PredictionlabelList = []

        for i in range(0, len(self.features[:self.val_i_end])):
            scores = []
            for speechmodel in self.speechmodels:
                scores.append(speechmodel.model.score(self.features[i]))
            id = scores.index(max(scores))
            self.m_PredictionlabelList.append(self.speechmodels[id].Class)
            print(str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) + " " + ":" +
                  self.speechmodels[id].label)

        self.get_accuracy(save_path=save_path)

    def predict(self, files):
        features = self._get_features(files, eval=True)
        Model_confidence = collections.namedtuple('model_prediction', ('name', 'score'))
        predicted_labels_confs = []

        for i in range(0, len(features)):
            file_scores_confs = []
            for speechmodel in self.speechmodels:
                score = speechmodel.model.score(features[i])
                label = speechmodel.label
                file_scores_confs.append(Model_confidence(name=label, score=score))
                file_scores_confs = sorted(file_scores_confs, key=itemgetter(1), reverse=True)
            predicted_labels_confs.append(file_scores_confs)

        return predicted_labels_confs

    # Calcuation of  mean ,entropy and relative entropy parameters
    '''Entropyvalues for the 3 hidden states and 100 samples'''

    def entropy_calculator(self, dataarray, meanvalues, sigmavalues):
        entropyvals = []
        for i in range(0, len(dataarray[0])):
            totallogpdf = 0
            entropy = 0
            for j in range(0, len(dataarray)):
                totallogpdf += sp.norm.logpdf(dataarray[j, i], meanvalues[i], sigmavalues[i])
                entropy = (-1 * totallogpdf) / len(dataarray)
            entropyvals.append(entropy)
        return entropyvals

    '''Relative Entropyvalues for the 6 columns of the given data and sampled values'''

    def relative_entropy_calculator(self, givendata, samplesdata, givendatasigmavals, sampledsigmavals,
                                    givendatameanvals, sampledmeanvals):

        absgivendatasigmavals = [abs(number) for number in givendatasigmavals]
        abssampleddatasigmavals = [abs(number) for number in sampledsigmavals]
        relativeentropyvals = []

        for i in range(0, len(givendata[0])):
            totallogpdf = 0
            relativeentropy = 0
            for j in range(0, len(givendata)):
                totallogpdf += (sp.norm.logpdf(samplesdata[j, i], sampledmeanvals[i],
                                               abssampleddatasigmavals[i]) - sp.norm.logpdf(givendata[j, i],
                                                                                            givendatameanvals[i],
                                                                                            absgivendatasigmavals[i]))
                relativeentropy = (-1 * totallogpdf) / float(len(givendata))
            relativeentropyvals.append(relativeentropy)
        return relativeentropyvals

    def calc_mean_entropy(self):

        for speechmodel in self.speechmodels:
            print("For GMMHMM with label:" + speechmodel.label)
            samplesdata, state_sequence = speechmodel.model.sample(n_samples=len(speechmodel.traindata))

            sigmavals = []
            meanvals = []

            for i in range(0, len(speechmodel.traindata[0])):
                sigmavals.append(np.mean(speechmodel.traindata[:, i]))
                meanvals.append(np.std(speechmodel.traindata[:, i]))

            sampledmeanvals = []
            sampledsigmavals = []

            for i in range(0, len(samplesdata[0])):
                sampledmeanvals.append(np.mean(samplesdata[:, i]))
                sampledsigmavals.append(np.std(samplesdata[:, i]))

            GivenDataEntropyVals = self.entropy_calculator(speechmodel.traindata, meanvals, meanvals)
            SampledValuesEntropyVals = self.entropy_calculator(samplesdata, sampledmeanvals, sampledsigmavals)
            RelativeEntropy = self.relative_entropy_calculator(speechmodel.traindata, samplesdata, sigmavals,
                                                               sampledsigmavals, meanvals, sampledmeanvals)

            print("MeanforGivenDataValues:")
            roundedmeanvals = np.round(meanvals, 3)
            print(str(roundedmeanvals))
            print("")

            print("EntropyforGivenDataValues:")
            roundedentropyvals = np.round(GivenDataEntropyVals, 3)
            print(str(roundedentropyvals))
            print("")

            print("MeanforSampleddatavalues:")
            roundedsampledmeanvals = np.round(sampledmeanvals, 3)
            print(str(roundedsampledmeanvals))
            print("")

            print("EntropyforSampledDataValues:")
            roundedsampledentvals = np.round(SampledValuesEntropyVals, 3)
            print(str(roundedsampledentvals))
            print("")

            print("RelativeEntopyValues:")
            roundedrelativeentvals = np.round(RelativeEntropy, 3)
            print(str(roundedrelativeentvals))
            print("")

    def pickle(self, filename):
        '''save model to file'''
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        '''read model from file'''
        with open(filename, 'rb') as f:
            return pickle.load(f)


class SpeechModel:
    def __init__(self, Class, label, m_num_of_HMMStates, m_num_of_mixtures, m_transmatPrior, m_startprobPrior,
                 m_covarianceType='diag', m_n_iter=10, n_features_traindata=6):
        self.traindata = np.zeros((0, n_features_traindata))
        self.Class = Class
        self.label = label
        self.model = hmm.GMMHMM(n_components=m_num_of_HMMStates, n_mix=m_num_of_mixtures,
                                transmat_prior=m_transmatPrior, startprob_prior=m_startprobPrior,
                                covariance_type=m_covarianceType, n_iter=m_n_iter)
