import pandas as pd
import numpy as np

"""
Stores three representations of the data:
- The original dataframe
- A model-friendly version of the dataframe (one-hot encoding, etc)
- A recourse-friendly version of the dataframe (one-hot encoding, columns processed, etc)
"""
class DataAdapter:
  def __init__(self, data_df, model_data, recourse_data):
    self.data_df = data_df
    self.model_data = model_data
    self.recourse_data = recourse_data
    self.X = self.recourse_data.X
    self.Y = self.recourse_data.Y

  def random_poi(self, df=None, label=-1):
    if df is None:
      df = self.recourse_data.df
    return df[df['Y'] == label].drop(columns=['Y']).sample(1)

  def filter_accuracy(self, model_scores, cutoff=0.7):
    df = self.recourse_data.df
    return df[model_scores >= cutoff]

  def filter_from_poi(self, poi, df, immutable_features, tolerance_dict):
    df = df.drop(labels=poi.index, axis=0)
    for feature in immutable_features:
      mask = None
      columns = self.recourse_data.get_feature_columns(feature)
      tol = 0
      if feature in tolerance_dict:
        tol = tolerance_dict[feature]
      mask = (np.abs(df[columns].values - poi[columns].values) <= tol).all(axis=1)
      df = df[mask]
    return df.drop(columns=self.recourse_data.get_feature_columns(immutable_features))

  def print_point(self, poi, columns_to_print=None): # lookup the data representation and print that
    if columns_to_print is not None:
      print(self.data_df.loc[poi.index, columns_to_print])
    else:
      print(self.data_df.loc[poi.index,:])

  def print_recourse(dir): # set missing values to 0; no change
    pass

  def data_representation(poi, points, immutable_features=None):
    recourse_df = self.reconstruct_data_from_poi(poi, points, immutable_features)
    data_df = self.recourse_data.recover_data_representation(poi, recourse_df)
    return data_df

  def model_representation(df):
    return self.model_data.convert(df)

  def reconstruct_data_from_poi(poi, points, immutable_features=None):
    columns, dropped_columns = self.recourse_data.recover_columns(immutable_features=immutable_features)
    new_points = np.empty((points.shape[0], poi.shape[1]))
    new_points[:,columns.shape[0]:] = points
    new_points[:,:columns.shape[0]] = poi[dropped_columns]
    df = pd.DataFrame(columns=np.concatenate([columns, dropped_columns]), data=new_points)
    return df
