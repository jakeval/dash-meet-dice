from data_handling.data_adapter import DataAdapter
from data_handling.data_classes import ModelData, RecourseData
import pandas as pd
import csv

class AdultDataAdapter(DataAdapter):
  def __init__(self, train_filename, test_filename, reformat=False):
    self.category_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    self.binary_columns = ['sex']
    if reformat:
        train_filename, test_filename = self.reformat_data(train_filename), self.reformat_data(test_filename, is_test=True)
    data_df, test_df = self.load_and_process_data(train_filename), self.load_and_process_data(test_filename)
    model_data = ModelData(data_df, test_df)
    recourse_data = RecourseData(data_df)
    super().__init__(data_df, model_data, recourse_data)

  def _convert_label(self, df):
    new_df = df.copy()
    new_df['income'] = df['income'].mask(df['income'] == '>50K', 1)
    new_df['income'] = new_df['income'].mask(df['income'] == '<=50K', -1)
    return new_df

  def reformat_data(self, filename, is_test=False):
    reader = None
    new_filename = f"{filename}.reformatted"
    column_names = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income".split(',')
    with open(filename, newline='') as file:
      reader = csv.reader(file, delimiter=',')
      with open(new_filename, 'w', newline='') as newfile:
        writer = csv.writer(newfile, delimiter=',')
        writer.writerow(column_names)
        for i, row in enumerate(reader):
          if is_test and i == 0:
            continue
          stripped_row = list(map(lambda s: s.strip(), row))
          if is_test:
            stripped_row = list(map(lambda s: s.rstrip('.'), stripped_row))
          writer.writerow(stripped_row)
    return new_filename

  def load_and_process_data(self, filename):
    df = None
    with open(filename) as f:
      df = pd.read_csv(f)

    df = df.drop(columns=['education', 'fnlwgt'])
    df = df.drop_duplicates()

    for c in df.columns:
      df = df.drop(index=df[df[c] == '?'].index)
    df = self._convert_label(df)

    return df
