import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.preprocessing, sklearn.metrics, sklearn.utils, sklearn.multiclass, sklearn.linear_model

features = pd.read_csv('data/fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('data/fma_metadata/tracks.csv', index_col=0, header=[0, 1])

CENS = 'chroma_cens'
CQT = 'chroma_cqt'
STFT = 'chroma_stft'
MFCC = 'mfcc'
RMSE = 'rmse'
BW = 'spectral_bandwidth'
CENT = 'spectral_centroid'
CONT = 'spectral_contrast'
ROLLOFF = 'spectral_rolloff'
TON = 'tonnetz'
ZCR = 'zcr'
all_features = [CENS,CQT,STFT,MFCC,RMSE,BW,CENT,CONT,ROLLOFF,TON,ZCR]

class LR(object):
    def __init__(self, tracks, features):
        self.tracks = tracks
        self.features = features
        self.small = tracks['set', 'subset'] <= 'small'
        self.training = tracks['set', 'split'] == 'training'
        self.validation = tracks['set', 'split'] == 'validation'
        self.testing = tracks['set', 'split'] == 'test'
    
    
    def datasplit(self, feature_array):
        # takes an array of features [MFCC, CONT]
        X_train_temp = self.features.loc[self.small & (self.training | self.validation), feature_array]
        X_test_temp = self.features.loc[self.small & self.testing, feature_array]
        y_train_temp = self.tracks.loc[self.small & (self.training | self.validation), ('track', 'genre_top')]
        y_test_temp = self.tracks.loc[self.small & self.testing, ('track', 'genre_top')]
        y_train = y_train_temp.dropna()
        y_test = y_test_temp.dropna()
        X_train = X_train_temp.drop(y_train_temp.drop(y_train.index).index)
        X_test = X_test_temp.drop(y_test_temp.drop(y_test.index).index)
        EXPERIMENTAL = self.tracks['track', 'genre_top'] == "Experimental"
        X_train = X_train.drop(X_train.loc[EXPERIMENTAL].index)
        y_train = y_train.drop(y_train.loc[EXPERIMENTAL].index)
        X_test = X_test.drop(X_test.loc[EXPERIMENTAL].index)
        y_test = y_test.drop(y_test.loc[EXPERIMENTAL].index)
        return skl.utils.shuffle(X_train, y_train, random_state=42), X_test, y_test

    # given parameters achieves the highest score
    def train(self, feature_array=all_features, solver='liblinear', penalty='l1', multi_class='ovr', C=0.072):
        (X_train, y_train), X_test, y_test = self.datasplit(feature_array)
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        
        # LR through OneVsOne scheme        
        base_lr = skl.linear_model.LogisticRegression(solver=solver,penalty=penalty,multi_class=multi_class,C=C, random_state=0)
        self.classifier = skl.multiclass.OneVsOneClassifier(base_lr)
        print(self.classifier)
        self.classifier.fit(X_train, y_train)        
        print("Logistic Regression OneVsOne(solver =", solver, ")")
        
        print("Training Report:", self.classifier.score(X_train,y_train))
        print(skl.metrics.classification_report(y_train, self.classifier.predict(X_train)))
        print()
        print("Test Report: Accuracy = ", self.classifier.score(X_test,y_test))
        print(skl.metrics.classification_report(y_test, self.classifier.predict(X_test)))
        
    def cross_validation(self,feature_array=all_features):
        (X_train, y_train), X_test, y_test = self.datasplit(feature_array)
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            
            tuned_parameters = {'C':10.0 ** -np.arange(1, 7), 'penalty':['l1'],
                                'solver':['liblinear','saga'], 'multi_class':['ovr'], 'random_state':[0]}
            clf = skl.model_selection.GridSearchCV(skl.linear_model.LogisticRegression(), tuned_parameters, cv=5, scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()          
