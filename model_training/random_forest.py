import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import math
import pandas as pd
import pickle
import json

class RandomForest:
    stats_dict = {'mean': 0, 'std': 1}

    def __init__(self, model, std, mean, pca):
        self.model = model
        self.mean = mean
        self.std = std
        self.pca = pca
    
    def predict(self, X):
        Xnew = self.process_data(X)
        return self.model.predict(Xnew)

    def positive_prob(self, X):
        Xnew = self.process_data(X)
        return self.model.positive_proba(Xnew)

    def process_data(self, X):
        X = np.array(X)

        X = (X - self.mean) / self.std

        pca = PCA()
        X = pca.fit_transform(X)
        return X

    def save_model(self):
        with open('saved_models/random_forest/stats.json', 'wb') as f:
            stats = np.vstack([self.mean, self.std])
            print(stats.shape)
            np.save(f, stats)
        with open('saved_models/random_forest/pca.pickle', 'wb') as f:
            pickle.dump(self.pca, f)
        with open('saved_models/random_forest/model.pickle', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model():
        mean = std = model = pca = None
        with open('saved_models/random_forest/stats.json', 'rb') as f:
            stats = np.load(f)
            mean = stats[RandomForest.stats_dict['mean']]
            std = stats[RandomForest.stats_dict['std']]
        with open('saved_models/random_forest/pca.pickle', 'rb') as f:
            pca = pickle.load(f)
        with open('saved_models/random_forest/model.pickle', 'rb') as f:
            model = pickle.load(f)
        return RandomForest(model, std, mean, pca)


def get_accuracy(y_pred, y_true):
    correct_count = y_pred[y_pred == y_true].shape[0]
    incorrect_count = y_pred[y_pred != y_true].shape[0]
    return correct_count / (correct_count + incorrect_count)

def train_random_forest(X, Y, X_test, Y_test):

    X = np.array(X)
    Y = np.array(Y).astype(np.float)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test).astype(np.float)

    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    pca = PCA()
    X = pca.fit_transform(X)
    X_test = pca.transform(X_test)

    n_list = [5, 10, 100, 1000]
    split = 0.01 # already checked for best value
    best_model = None
    best_score = -math.inf
    for n in n_list:
        rf = RandomForestClassifier(n_estimators=n, min_samples_split=split)
        rf.fit(X, Y)
        y_pred = rf.predict(X_test)
        accuracy = get_accuracy(y_pred, Y_test)
        print(f"\t accuracy: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_model = rf
    print(best_score)
    return RandomForest(best_model, std, mean, pca)
