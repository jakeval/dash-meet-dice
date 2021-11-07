from sklearn.decomposition import PCA
import dice_ml
from dice_ml.utils import helpers
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


dataset = helpers.load_adult_income_dataset()

target = dataset['income']

datasetX = dataset.drop('income', axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    datasetX,
    target,
    test_size=0.2,
    random_state=0,
    stratify=target
)

model = load('./saved_models/random_forest/random_forest.joblib')
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
backend = 'sklearn'
m = dice_ml.Model(model=model, backend=backend)
exp_kd = dice_ml.Dice(d, m, method="random")

pca = None
transformer = None

def get_pca_2d():
    global transformer
    transformer = get_transformer(x_train)

    data = transformer.fit_transform(x_train).toarray()
    global pca
    pca = PCA(2)
    pca_data = pca.fit_transform(data)
    df = x_train.copy()
    df['xpca'] = pca_data[:,0]
    df['ypca'] = pca_data[:,1]
    df['income'] = y_train.copy()
    return df

def pca_process(df):
    global pca
    global transformer

    data = transformer.transform(df).toarray()
    pca_data = pca.transform(data)
    dfnew = df.copy()
    dfnew['xpca'] = pca_data[:,0]
    dfnew['ypca'] = pca_data[:,1]
    return dfnew

def get_accuracy(y_pred, y_true):
    correct_count = y_pred[y_pred == y_true].shape[0]
    incorrect_count = y_pred[y_pred != y_true].shape[0]
    return correct_count / (correct_count + incorrect_count)

def get_explanations(poi):
    dice_exp = exp_kd.generate_counterfactuals(poi.drop('income', axis=1), total_CFs=3, desired_class="opposite", verbose=False)
    return dice_exp.cf_examples_list[0].final_cfs_df

def get_transformer(x):
        numerical = ['age', 'hours_per_week']
        categorical = x.columns.difference(numerical)

        numeric_transformer = Pipeline(steps=[
            ('scalar', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical),
                ('cat', categorical_transformer, categorical)])
        
        return transformations

if __name__ == '__main__':
    save = False
    if save:
        print("Train model...")
        transformations = get_transformer(x_train)

        clf = Pipeline(steps=[
            ('preprocessor', transformations),
            ('classifier', RandomForestClassifier())])

        model = clf.fit(x_train, y_train)

        print("Finished training. Predict and save...")
        y_pred = model.predict(x_test)
        print(get_accuracy(y_pred, y_test))
        
        dump(model, './saved_models/random_forest/random_forest.joblib')
