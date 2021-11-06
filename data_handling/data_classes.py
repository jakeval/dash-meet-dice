import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class ModelData:
  def __init__(self, train_df, test_df):
    self.binary_columns = ['sex']
    self.category_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    self.train_df, self.test_df, self.le_dict, self.ohe = self.process_data(train_df.copy(), test_df.copy())

  def process_data(self, train_df, test_df):
    le_dict = {}
    for column in self.binary_columns:
      le = LabelEncoder()
      le.fit(train_df[column])
      train_df[column] = le.transform(train_df[column])
      test_df[column] = le.transform(test_df[column])
      le_dict[column] = le

    ohe = OneHotEncoder().fit(train_df[self.category_columns])
    for df in [train_df, test_df]:
      new_data = ohe.transform(df[self.category_columns]).toarray()
      new_columns = ohe.get_feature_names_out(self.category_columns)
      df[new_columns] = new_data
    train_df = train_df.drop(columns=self.category_columns)
    test_df = test_df.drop(columns=self.category_columns)

    train_df = train_df.rename(columns={'income': 'Y'})
    test_df = test_df.rename(columns={'income': 'Y'})

    return train_df, test_df, le_dict, ohe

  def convert(self, df):
    model_df = df.copy()
    for column, le in self.le_dict.items():
      model_df[column] = le.transform(df[column])
    
    new_data = self.ohe.transform(df[self.category_columns]).toarray()
    new_columns = self.ohe.get_feature_names_out(self.category_columns)
    model_df[new_columns] = new_data
    model_df = model_df.drop(columns=self.category_columns)

    return model_df


class RecourseData:
  def __init__(self, df):
    self.original_df = df.copy()
    df = df.copy()
    self.dropped_columns = ['native-country', 'relationship']
    self.binary_columns = ['marital-status', 'sex']
    self.category_columns = ['workclass', 'occupation', 'race']
    df_X = df.drop(columns=['income'])
    self.mean, self.std = df_X.mean(numeric_only=True), df_X.std(numeric_only=True)
    self.normalized_columns = self.mean.index
    df[self.normalized_columns] = (df[self.normalized_columns] - self.mean)/self.std
    self.df, self.le_dict, self.ohe_dict = self.process_data(df)
    self.X = self.df.drop(columns=['Y'])
    self.Y = self.df['Y']

  def process_data(self, df):
    recourse_df = df.copy()
    recourse_df = recourse_df.drop(columns=self.dropped_columns)

    recourse_to_data_categories = {
        'Single': ['Never-married', 'Divorced', 'Separated', 'Widowed'],
        'Married': ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    }
    recourse_df['marital-status'] = self._recategorize_column(recourse_df['marital-status'], recourse_to_data_categories)

    le_dict = {}
    for column in self.binary_columns:
      le = LabelEncoder()
      recourse_df[column] = le.fit_transform(recourse_df[column])
      le_dict[column] = le
    
    ohe_dict = {}
    for column in self.category_columns:
      column_df = recourse_df.loc[:, recourse_df.columns == column]
      ohe = OneHotEncoder().fit(column_df)
      new_data = ohe.transform(column_df).toarray()
      new_columns = ohe.get_feature_names_out([column])
      recourse_df[new_columns] = new_data
      ohe_dict[column] = ohe
    recourse_df = recourse_df.drop(columns=self.category_columns)

    recourse_df = recourse_df.rename(columns={'income': 'Y'})
    
    return recourse_df, le_dict, ohe_dict

  def get_feature_columns(self, features):
    if type(features) == str:
      features = [features]
    columns = []
    for feature in features:
      if feature in self.ohe_dict:
        dummy_columns = self.ohe_dict[feature].get_feature_names_out([feature])
        columns = columns + dummy_columns
      else:
        columns.append(feature)
    return columns

  def recover_columns(self, immutable_features=None):
    all_columns = self.X.columns
    dropped_columns = []
    if immutable_features is not None:
      dropped_columns = self.ohe.get_feature_names_out(immutable_features)
    retained_columns = all_columns[~all_columns.isin(dropped_columns)]
    return retained_columns, dropped_columns

  def recover_data_representation(self, poi, df):
    recourse_df = df.copy()
    for column, le in self.le_dict:
      recourse_df[column] = le.inverse_transform(recourse_df[column])
    encoded_columns = self.ohe.get_feature_names_out(self.category_columns)
    recourse_df[self.category_columns] = self.ohe.inverse_transform(recourse_df[encoded_columns])
    recourse_df = recourse_df * self.std + self.mean

    poi_recovered_data = self.original_df.loc[poi.index,self.original_df.columns == self.dropped_columns]
    recourse_df[self.dropped_columns] = poi_recovered_data
    return recourse_df

  def _recategorize_column(self, column, inverse_column_map):
    new_column = column.copy()
    for key, val_list in inverse_column_map.items():
      for val in val_list:
        new_column = np.where(new_column == val, key, new_column)
    return new_column
