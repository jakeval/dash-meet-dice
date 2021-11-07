from scipy.sparse.construct import random
from data_handling.adult_data_adapter import AdultDataAdapter
from sklearn.decomposition import PCA
from model_training import random_forest
import pickle
import json
import numpy as np

da = AdultDataAdapter('../data/adult.data.reformatted', '../data/adult.test.reformatted', reformat=False)

def get_pca_2d():
    pca = PCA(2)
    pca_data = pca.fit_transform(da.X.values)
    df = da.data_df.copy()
    df['xpos'] = pca_data[:,0]
    df['ypos'] = pca_data[:,1]
    df['income'] = np.where(df['income'].values == 1, '>50', '<=50')
    return df

def get_pca_3d():
    pca = PCA(3)
    pca_data = pca.fit_transform(da.X.values)
    df = da.recourse_data.df.copy()
    df['xpos'] = pca_data[:,0]
    df['ypos'] = pca_data[:,1]
    df['zpos'] = pca_data[:,2]
    return df

def get_printable_poi():
    poi = da.random_poi()
    return da.data_representation(poi)

def get_printable_points(n):
    return da.data_df.sample(n)

def load_model():
    return random_forest.RandomForest.load_model()

def train_model():
    train = da.model_data.train_df
    test = da.model_data.test_df
    X = train.drop(columns=['Y'])
    Y = train['Y']

    X_test = test.drop(columns=['Y'])
    Y_test = test['Y']

    return random_forest.train_random_forest(X, Y, X_test, Y_test)


if __name__ == '__main__':
    save = False
    if save:
        print("Train model...")
        model = train_model()
        model.save_model()
    else:
        model = random_forest.RandomForest.load_model()
        print(model.model)
        print(model.std)