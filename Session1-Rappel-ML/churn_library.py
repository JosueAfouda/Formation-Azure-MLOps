"""
This is the churn_library.py python file that we used to find customers who are likely to churn
The execution of this file will produce artefacts in images and models folders.
Date: April 02, 2023
Author: Josue AFOUDA
"""
# import libraries
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


# Pour la journalisation
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Importation des données


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].apply(lambda val: 0 if val == "No" else 1)

    return df


def data_spliting(df):
    '''
    input:
              df: pandas dataframe

    output:
              train: training dataframe
              validate: validation dataframe
              test: test dataframe
    '''
    train, test = train_test_split(
        df, test_size=0.3, random_state=123, stratify=df['Churn']
    )
    test, validate = train_test_split(
        test, test_size=0.5, random_state=123, stratify=test['Churn']
    )

    # Enregistrement des différents ensembles de données
    train.to_csv('./data/train.csv', index=False)
    validate.to_csv('./data/validation.csv', index=False)
    test.to_csv('./data/test.csv', index=False)

    X_train, X_val = train.drop(
        'Churn', axis=1), validate.drop(
        'Churn', axis=1)
    y_train, y_val = train['Churn'], validate['Churn']

    return train, X_train, y_train, X_val, y_val


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: a pandas dataframe

    output:
            None
    '''
    df_copy = df.copy()

    list_columns = df_copy.columns.to_list()

    list_columns.append('Heatmap')

    df_corr = df_copy.corr(numeric_only=True)

    for column_name in list_columns:
        plt.figure(figsize=(10, 6))
        if column_name == 'Heatmap':
            sns.heatmap(
                df_corr,
                mask=np.triu(np.ones_like(df_corr, dtype=bool)),
                center=0, cmap='RdBu', linewidths=1, annot=True,
                fmt=".2f", vmin=-1, vmax=1
            )
        else:
            if df[column_name].dtype != 'O':
                df[column_name].hist()
            else:
                sns.countplot(data=df, x=column_name)
        plt.savefig("images/eda/" + column_name + ".jpg")
        plt.close()


def classification_report_image(y_train,
                                y_train_preds,
                                y_val,
                                y_val_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_train_preds: training predictions from logistic regression
            y_val: validation response values
            y_val_preds: validation predictions from logistic regression
    output:
             None
    '''
    class_reports_dico = {
        "Logistic Regression train results": classification_report(
            y_train,
            y_train_preds),
        "Logistic Regression validation results": classification_report(
            y_val,
            y_val_preds)}

    for title, report in class_reports_dico.items():
        plt.rc('figure', figsize=(7, 3))
        plt.text(
            0.2, 0.3, str(report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.title(title, fontweight='bold')
        plt.savefig("images/results/" + title + ".jpg")
        plt.close()


# Fonction pour régler le problème de la colonne 'TotalCharges'
# Fonction pour régler le problème de la colonne 'TotalCharges'
def convert_totalcharges(X):
    '''
    Convertion of TotalCharges column to numeric
    input:
            X: dataframe of features
    output:
            a numpy array
    '''
    # X : dataframe
    Z = X.copy()
    Z['TotalCharges'] = pd.to_numeric(Z['TotalCharges'], errors='coerce')
    return Z.values


def build_pipeline():
    '''
    build a pipeline that contains preprocessing steps and an estimator
    input:
            None
    output:
            a scikit-learn pipeline object
    '''
    numeric_features = [
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    categorical_features = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod'
    ]

    # Pipeline de prétraitement des variables indépendantes numériques
    numeric_transformer = Pipeline(
        steps=[('convert', FunctionTransformer(convert_totalcharges)),
               ('imputer', SimpleImputer(strategy='median')),
               ('scaler', StandardScaler())]
    )

    # Pipeline de prétraitement des variables indépendantes qualitatives
    categorical_transformer = Pipeline(
        steps=[
            ('onehotencoder',
             OneHotEncoder(
                 sparse_output=False,
                 handle_unknown='ignore'))])

    # Combinaison des deux précédents pipelines en un seul
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric',
             numeric_transformer,
             numeric_features),
            ('categorical',
             categorical_transformer,
             categorical_features)])

    # Pipeline de modélisation
    pipeline_model = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('logreg', LogisticRegression(solver='newton-cg',
                                             random_state=123,
                                             max_iter=2000,
                                             C=5.0,
                                             penalty='l2'))]
    )

    return pipeline_model


def train_models(X_train, X_val, y_train, y_val):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_val: X validation data
              y_train: y training data
              y_val: y validation data
    output:
              None
    '''
    # Formation du modèle
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Predictions
    y_train_preds_lr = model.predict(X_train)
    y_val_preds_lr = model.predict(X_val)

    # ROC curves image
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.savefig("images/results/roc_curve.jpg")
    plt.close()

    # Clasification reports images
    classification_report_image(
        y_train,
        y_train_preds_lr,
        y_val,
        y_val_preds_lr)

    # Sauvegarde du modèle
    joblib.dump(model, './models/logreg_model.pkl')


def main():
    logging.info("Importation des donnees...")
    raw_data = import_data("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    logging.info("Importation des donnees : SUCCES")

    logging.info("Division des donnees...")
    train_data, Xtrain, ytrain, Xval, yval = data_spliting(raw_data)
    logging.info("Division des donnees : SUCCESS")

    logging.info("Analyse exploratoire des donnees...")
    perform_eda(train_data)
    logging.info("Analyse exploratoire des donnees : SUCCES")

    logging.info("Formation du modele...")
    train_models(Xtrain, Xval, ytrain, yval)
    logging.info("Formation du modele : SUCCES")


if __name__ == "__main__":
    print("Execution en cours...")
    main()
    print("Fin de l'Execution ave succès")
