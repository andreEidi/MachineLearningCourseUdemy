import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
# ExtraTree é um classificador mas funciona selecionando atributos
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Extra tree - Random Forest x Extra Trees Classifier: https://www.thekerneltrip.com/statistics/random-forest-vs-extra-tree/


def getData():
    base_census = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/census.csv')

    return base_census


def printInfoColumms(base_census):
    # Verifica as colunas exceto a última
    colunas = base_census.columns[:-1]
    # Verifica as colunas exceto as duas últimas...
    colunas2 = base_census.columns[:-2]
    print("Colunas: ", colunas)
    print("--------------------------------------------------")
    print("Colunas2: ", colunas2)
    print("--------------------------------------------------")
    return colunas


def selectFeatures(base_census):
    X_census = base_census.iloc[:, 0:14].values
    y_census = base_census.iloc[:, 14].values
    print("X_census: \n", X_census)
    print("--------------------------------------------------")
    print("y_census: \n", y_census)
    print("--------------------------------------------------")
    return X_census, y_census


def labelMap(X_census):
    # transformar categoricos em numericos
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

    print("X_census - LabelEncoder \n: ", X_census)
    print("--------------------------------------------------")
    return X_census


def scaler(X_census):
    # Normalizador (Nas outras aulas usamos o standarscale)
    scaler = MinMaxScaler()
    X_census_scaler = scaler.fit_transform(X_census)
    print("X_census_scaler \n", X_census_scaler)
    print("--------------------------------------------------")
    return X_census_scaler


def modelSelection(X_census_scaler, y_census):
    selecao = ExtraTreesClassifier()
    selecao.fit(X_census_scaler, y_census)
    return selecao


def checkColummImportances(selecao):
    importancias = selecao.feature_importances_
    print("importances: \n", importancias)
    return importancias


def selectFeaturesExtra(importancias, X_census):
    indices = []
    # retorna os indices de features cujas importancias sejam maiores que um certo valor
    for i in range(len(importancias)):
        # print(i)
        if importancias[i] >= 0.029:
            indices.append(i)

    X_census_extra = X_census[:, indices]
    return X_census_extra


def encoder(X_census_extra):

    onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [
                                       1, 3, 5, 6, 7])], remainder='passthrough')
    X_census_extra = onehotencorder.fit_transform(X_census_extra).toarray()
    return X_census_extra


def slipt(X_census_extra, y_census):
    X_census_treinamento_extra, X_census_teste_extra, y_census_treinamento_extra, y_census_teste_extra = train_test_split(
        X_census_extra, y_census, test_size=0.15, random_state=0)
    return X_census_treinamento_extra, X_census_teste_extra, y_census_treinamento_extra, y_census_teste_extra


def model(X_census_treinamento_extra, y_census_treinamento_extra):
    from sklearn.ensemble import RandomForestClassifier
    random_forest_extra = RandomForestClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    random_forest_extra.fit(X_census_treinamento_extra,
                            y_census_treinamento_extra)
    return random_forest_extra


def metrics(random_forest_extra, X_census_teste_extra, y_census_teste_extra):
    previsoes = random_forest_extra.predict(X_census_teste_extra)
    print("Accuracy: \n", accuracy_score(y_census_teste_extra, previsoes))


def main():
    base_census = getData()
    colunas = printInfoColumms(base_census)
    X_census, y_census = selectFeatures(base_census)
    X_census = labelMap(X_census)
    X_census_scaler = scaler(X_census)
    selecao = modelSelection(X_census_scaler, y_census)
    importancias = checkColummImportances(selecao)
    X_census_extra = selectFeaturesExtra(importancias, X_census)
    X_census_extra = encoder(X_census_extra)
    X_census_treinamento_extra, X_census_teste_extra, y_census_treinamento_extra, y_census_teste_extra = slipt(
        X_census_extra, y_census)
    random_forest_extra = model(
        X_census_treinamento_extra, y_census_treinamento_extra)
    metrics(random_forest_extra, X_census_teste_extra, y_census_teste_extra)


main()
