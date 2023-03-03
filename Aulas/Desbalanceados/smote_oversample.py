import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def getData():
    base_census = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/census.csv')

    return base_census


def printInfo(base_census):
    print("income: ", np.unique(base_census['income'], return_counts=True))
    print("------------------------------------------------------------------")
    sns.countplot(x=base_census['income'])
    plt.show()


def selectFeatures(base_census):
    X_census = base_census.iloc[:, 0:14].values
    y_census = base_census.iloc[:, 14].values
    return X_census, y_census


def labelMap(X_census):

    label_encoder_workclass = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_marital = LabelEncoder()
    label_encoder_occupation = LabelEncoder()
    label_encoder_relationship = LabelEncoder()
    label_encoder_race = LabelEncoder()
    label_encoder_sex = LabelEncoder()
    label_encoder_country = LabelEncoder()

    X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
    X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
    X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
    X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
    X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
    X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
    X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
    X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

    print("X_census: ", X_census)
    print("------------------------------------------------------------------")
    return X_census


def oversample(X_census, y_census):

    smote = SMOTE(sampling_strategy='minority')
    X_over, y_over = smote.fit_resample(
        X_census, y_census)  # Atualizado 20/05/2022
    print("Original: ", np.unique(y_census, return_counts=True))
    print("Aplicado algoritmo: ", np.unique(y_over, return_counts=True))
    print("------------------------------------------------------------------")
    return X_over, y_over


def trainSplit(X_over, y_over):

    X_census_treinamento_over, X_census_teste_over, y_census_treinamento_over, y_census_teste_over = train_test_split(
        X_over, y_over, test_size=0.15, random_state=0)
    return X_census_treinamento_over, X_census_teste_over, y_census_treinamento_over, y_census_teste_over


def model(X_census_treinamento_over, y_census_treinamento_over):
    random_forest_census = RandomForestClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    random_forest_census.fit(X_census_treinamento_over,
                             y_census_treinamento_over)
    return random_forest_census


def metrics(random_forest_census, X_census_teste_over, y_census_teste_over):
    from sklearn.metrics import accuracy_score, classification_report
    previsoes = random_forest_census.predict(X_census_teste_over)
    print("Accuracy: ", accuracy_score(y_census_teste_over, previsoes))
    print("Report: \n", classification_report(y_census_teste_over, previsoes))


def main():
    base_census = getData()
    printInfo(base_census)
    X_census, y_census = selectFeatures(base_census)
    X_census = labelMap(X_census)
    X_over, y_over = oversample(X_census, y_census)
    X_census_treinamento_over, X_census_teste_over, y_census_treinamento_over, y_census_teste_over = trainSplit(
        X_over, y_over)
    random_forest_census = model(
        X_census_treinamento_over, y_census_treinamento_over)
    metrics(random_forest_census, X_census_teste_over, y_census_teste_over)


main()
