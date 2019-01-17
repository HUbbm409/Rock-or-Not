import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.metrics import confusion_matrix
import itertools

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

ROCK = tracks['track', 'genre_top'] == "Rock"
ELECTRONIC = tracks['track', 'genre_top'] == "Electronic"
CLASSICAL = tracks['track', 'genre_top'] == "Classical"
HIPHOP = tracks['track', 'genre_top'] == "Hip-Hop"
FOLK = tracks['track', 'genre_top'] == "Folk"
INSTRUMENTAL = tracks['track', 'genre_top'] == "Instrumental"
OLD = tracks['track', 'genre_top'] == "Old-Time / Historic"
INTERNATIONAL = tracks['track', 'genre_top'] == "International"
EXPERIMENTAL = tracks['track', 'genre_top'] == "Experimental"
POP = tracks['track', 'genre_top'] == "Pop"
JAZZ = tracks['track', 'genre_top'] == "Jazz"
SPOKEN = tracks['track', 'genre_top'] == "Spoken"
COUNTRY = tracks['track', 'genre_top'] == "Country"
SOUL = tracks['track', 'genre_top'] == "Soul-RnB"
BLUES = tracks['track', 'genre_top'] == "Blues"
EASY = tracks['track', 'genre_top'] == "Easy Listening"

all_genres = [ROCK,ELECTRONIC,CLASSICAL,HIPHOP,FOLK,INSTRUMENTAL,OLD,INTERNATIONAL,POP,JAZZ,SPOKEN,COUNTRY,SOUL,BLUES,EASY]
top_eight = all_genres[0:7]
top_four = all_genres[0:4]

class SVM(object):
    def __init__(self,tracks,features):
        self.tracks = tracks
        self.features = features    
    
    def datasplit(self):        
        small = self.tracks['set', 'subset'] <= 'small'
        training = self.tracks['set', 'split'] == 'training'
        validation = self.tracks['set', 'split'] == 'validation'
        testing = self.tracks['set', 'split'] == 'test'
        
        # drop other features
        X_train_temp = self.features.loc[small & (training | validation), self.feature_array]
        X_test_temp = self.features.loc[small & testing, self.feature_array]
        y_train_temp = self.tracks.loc[small & (training | validation), ('track', 'genre_top')]
        y_test_temp = self.tracks.loc[small & testing, ('track', 'genre_top')]
        
        # drop na
        y_train = y_train_temp.dropna()
        y_test = y_test_temp.dropna()
        X_train = X_train_temp.drop(y_train_temp.drop(y_train.index).index)
        X_test = X_test_temp.drop(y_test_temp.drop(y_test.index).index)
        
        genres = self.genre_array[0]
        for i in self.genre_array[1::]: genres = np.logical_or(genres, i)
            
        X_train = X_train.loc[genres]
        y_train = y_train.loc[genres]
        X_test = X_test.loc[genres]
        y_test = y_test.loc[genres]
                
        return skl.utils.shuffle(X_train, y_train, random_state=42), X_test, y_test

    def train(self, feature_array=[MFCC,CONT,CENT], genre_array=all_genres):
        self.feature_array = feature_array
        self.genre_array = genre_array
        (X_train, y_train), X_test, y_test = self.datasplit()
        self.scaler = skl.preprocessing.StandardScaler(copy=False)
        self.scaler.fit_transform(X_train)
        self.scaler.transform(X_test)
        
        self.classifier = skl.svm.SVC(kernel='rbf', C=1.2).fit(X_train, y_train)
        print(self.classifier)
        
        print("Training Report")
        print("Train Accuracy: ", self.classifier.score(X_train, y_train))
        y_pred = self.classifier.predict(X_train)
        print(sklearn.metrics.classification_report(y_train, y_pred))
        self.conf_mat(y_train, y_pred, "TRAIN")
        
        print()
        print("Test Report")
        print("Test Accuracy: ", self.classifier.score(X_test, y_test))
        y_pred = self.classifier.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred))
        self.conf_mat(y_test, y_pred, "TEST")
        
        
    def cv(self, feature_array=[MFCC], genre_count=15):
        self.feature_array = feature_array
        self.genre_array = genre_array
        (X_train, y_train), X_test, y_test = self.datasplit()
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            self.classifier = skl.model_selection.GridSearchCV(skl.svm.SVC(), tuned_parameters, cv=3, scoring='%s_macro' % score)
            self.classifier.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(self.classifier.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = self.classifier.cv_results_['mean_test_score']
            stds = self.classifier.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, self.classifier.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            
            print("\nDetailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.\n")
            y_true, y_pred = y_test, self.classifier.predict(X_test)
            print(sklearn.metrics.classification_report(y_true, y_pred))
            self.conf_mat(y_test, y_pred, "")
            print()
            
    def conf_mat(self, y_test, y_pred, title, cmap='binary'):
        
        class_names = y_test.unique()
        plt.figure(figsize=(10,10))

        cnf_matrix = confusion_matrix(y_test, y_pred,labels=class_names)
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(class_names))
        plt.title(title, fontsize=20)
        fmt = '.2f'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),horizontalalignment="center",color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)

        plt.colorbar(fraction=0.045)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=15)
        plt.yticks(tick_marks, class_names, fontsize=15)

        plt.tight_layout()
        plt.savefig(title + '.jpeg')
        plt.show()
