import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.preprocessing, sklearn.metrics, sklearn.utils, sklearn.multiclass, sklearn.linear_model

features = pd.read_csv('features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])

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

class NN(object):
    def __init__(self,tracks,features):
        self.tracks = tracks
        self.features = features
        self.small = tracks['set', 'subset'] <= 'small'
        self.training = tracks['set', 'split'] == 'training'
        self.validation = tracks['set', 'split'] == 'validation'
        self.testing = tracks['set', 'split'] == 'test'
    
    
    def datasplit(self,feature_array):
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
    def train(self, feature_array=all_features, layer_tuple=(103,15), solver='adam', activation='relu'):
        (X_train, y_train), X_test, y_test = self.datasplit(feature_array)
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        self.classifier = skl.neural_network.MLPClassifier(early_stopping=True, activation=activation,
                                                           alpha=3e-05, verbose=1,
                                                           hidden_layer_sizes=layer_tuple, nesterovs_momentum= True,
                                                           solver=solver, random_state=1, max_iter=20).fit(X_train,y_train)
        print(self.classifier)
        print("Training Report: ", self.classifier.score(X_train,y_train))
        print(skl.metrics.classification_report(y_train, self.classifier.predict(X_train)))
        print()
        print("Test Report: ", self.classifier.score(X_test,y_test))
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
            
            tuned_parameters = {'alpha':10.0 ** -np.arange(1, 7), 'solver':['adam'], 'hidden_layer_sizes':[(250,15),(240,15),(260,15)], 'activation':['logistic']}
            clf = skl.model_selection.GridSearchCV(skl.neural_network.MLPClassifier(), tuned_parameters, cv=3, scoring='%s_macro' % score)
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
            print(skl.metrics.classification_report(y_true, y_pred))
            print()          
            
    def monitor(self, feature_array=all_features, layer_tuple=(103,15), activation="relu",solver="adam"):
        (X_train, y_train), X_test, y_test = self.datasplit(feature_array)
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        self.classifier = skl.neural_network.MLPClassifier(hidden_layer_sizes=layer_tuple,
                                                           alpha=3e-05, beta_1=0.99,
                                                           nesterovs_momentum=True,
                                                           learning_rate_init=0.0008996,
                                                           random_state=1, max_iter=1, warm_start=True)
        print(self.classifier)
        train_acc = []
        test_acc = []
        loss = []
        for i in range(15):
            self.classifier.partial_fit(X_train, y_train, np.unique(y_test))
            train_acc.append(self.classifier.score(X_train,y_train))
            test_acc.append(self.classifier.score(X_test,y_test))
            loss.append(self.classifier.loss_)
        print(activation)
        print(solver)
        print("Training Accuracy = ", train_acc[test_acc.index(max(test_acc))])
        print("Test Accuracy = ", max(test_acc))
        print("Loss = ", loss[test_acc.index(max(test_acc))])
        print("Epoch = ", test_acc.index(max(test_acc)))
        plt.plot(train_acc, color="r",label="Training Accuracy")
        plt.plot(test_acc, color="b",label="Test Accuracy")
        plt.legend()
        plt.show()
        
        plt.plot(loss)
        plt.show()
        return test_acc
