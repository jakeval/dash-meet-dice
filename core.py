from data_handling.adult_data_adapter import AdultDataAdapter
from sklearn.decomposition import PCA
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