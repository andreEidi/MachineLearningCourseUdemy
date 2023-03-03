import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import TomekLinks
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


def undersample(X_census, y_census):

    tl = TomekLinks(sampling_strategy='all')
    # Apaga tanto dados do minoritario quanto do majoritario (all)
    # Para apagar só o majoritario (majority)
    X_under, y_under = tl.fit_resample(X_census, y_census)
    print("Original: ", np.unique(y_census, return_counts=True))
    print("Aplicado algoritmo: ", np.unique(y_under, return_counts=True))
    print("------------------------------------------------------------------")
    return X_under, y_under


def encoder(X_under):

    onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [
                                       1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
    X_census = onehotencorder.fit_transform(X_under).toarray()
    print("X_census: ", X_census)
    print("------------------------------------------------------------------")
    print("X_under: ", X_under)
    print("------------------------------------------------------------------")
    return X_under


def trainSplit(X_under, y_under):

    X_census_treinamento_under, X_census_teste_under, y_census_treinamento_under, y_census_teste_under = train_test_split(
        X_under, y_under, test_size=0.15, random_state=0)
    return X_census_treinamento_under, X_census_teste_under, y_census_treinamento_under, y_census_teste_under


def model(X_census_treinamento_under, y_census_treinamento_under):
    # 84.70% com os dados originais

    random_forest_census = RandomForestClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    random_forest_census.fit(X_census_treinamento_under,
                             y_census_treinamento_under)
    return random_forest_census


def metrics(random_forest_census, X_census_teste_under, y_census_teste_under):
    from sklearn.metrics import accuracy_score, classification_report
    previsoes = random_forest_census.predict(X_census_teste_under)
    print("Accuracy: ", accuracy_score(y_census_teste_under, previsoes))
    print("Report: \n", classification_report(y_census_teste_under, previsoes))


def main():
    base_census = getData()
    printInfo(base_census)
    X_census, y_census = selectFeatures(base_census)
    X_census = labelMap(X_census)
    X_under, y_under = undersample(X_census, y_census)
    X_under = encoder(X_under)
    X_census_treinamento_under, X_census_teste_under, y_census_treinamento_under, y_census_teste_under = trainSplit(
        X_under, y_under)
    random_forest_census = model(
        X_census_treinamento_under, y_census_treinamento_under)
    metrics(random_forest_census, X_census_teste_under, y_census_teste_under)


main()
