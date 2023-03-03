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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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


def checkVariance(X_census_scaler):
    for i in range(X_census_scaler.shape[1]):
        print(
            f"Para a feature {i} temos uma variância de: {X_census_scaler[:,i].var()}")
    print("--------------------------------------------------")


def selectFeaturesBasedVariance(X_census_scaler, base_census, colunas):
    # variancia pequena pode indicar que o dado não é tão relevante (0.05 foi um dos menores valores de variancia para a base )
    selecao = VarianceThreshold(threshold=0.05)
    X_census_variancia = selecao.fit_transform(X_census_scaler)
    print("selecao.variances_: \n", selecao.variances_)
    print("--------------------------------------------------")
    # onde a variancia foi maior que o threshold
    indices = np.where(selecao.variances_ > 0.05)
    print("Selecionou-se as features cuja variancia era tal como o especificado, sendo: ", indices)
    print("--------------------------------------------------")
    print("Colunas selecionadas são: ", colunas[indices])
    print("--------------------------------------------------")

    base_census_variancia = base_census.drop(columns=['age', 'workclass', 'final-weight',
                                                      'education-num', 'race', 'capital-gain',
                                                      'capital-loos', 'hour-per-week',
                                                      'native-country'], axis=1)
    return base_census_variancia


def separateAxis(base_census_variancia):
    X_census_variancia = base_census_variancia.iloc[:, 0:5].values
    y_census_variancia = base_census_variancia.iloc[:, 5].values
    return X_census_variancia, y_census_variancia


def newLabelEncoder(X_census_variancia):

    label_encoder_education = LabelEncoder()
    label_encoder_marital = LabelEncoder()
    label_encoder_occupation = LabelEncoder()
    label_encoder_relationship = LabelEncoder()
    label_encoder_sex = LabelEncoder()

    X_census_variancia[:, 0] = label_encoder_education.fit_transform(
        X_census_variancia[:, 0])
    X_census_variancia[:, 1] = label_encoder_marital.fit_transform(
        X_census_variancia[:, 1])
    X_census_variancia[:, 2] = label_encoder_occupation.fit_transform(
        X_census_variancia[:, 2])
    X_census_variancia[:, 3] = label_encoder_relationship.fit_transform(
        X_census_variancia[:, 3])
    X_census_variancia[:, 4] = label_encoder_sex.fit_transform(
        X_census_variancia[:, 4])
    return X_census_variancia


def encoder(X_census_variancia):

    onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [
                                       0, 1, 2, 3, 4])], remainder='passthrough')
    X_census_variancia = onehotencorder.fit_transform(
        X_census_variancia).toarray()
    return X_census_variancia


def split(X_census_variancia, y_census_variancia):
    X_census_treinamento_var, X_census_teste_var, y_census_treinamento_var, y_census_teste_var = train_test_split(
        X_census_variancia, y_census_variancia, test_size=0.15, random_state=0)
    return X_census_treinamento_var, X_census_teste_var, y_census_treinamento_var, y_census_teste_var


def model(X_census_treinamento_var, y_census_treinamento_var):

    random_forest_var = RandomForestClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    random_forest_var.fit(X_census_treinamento_var, y_census_treinamento_var)
    return random_forest_var


def metrics(random_forest_var, X_census_teste_var, y_census_teste_var):

    previsoes = random_forest_var.predict(X_census_teste_var)
    print("Accuracy: ", accuracy_score(y_census_teste_var, previsoes))
    print("--------------------------------------------------")


def main():
    base_census = getData()
    colunas = printInfoColumms(base_census)
    X_census, y_census = selectFeatures(base_census)
    X_census = labelMap(X_census)
    X_census_scaler = scaler(X_census)
    checkVariance(X_census_scaler)
    base_census_variancia = selectFeaturesBasedVariance(
        X_census_scaler, base_census, colunas)
    X_census_variancia, y_census_variancia = separateAxis(
        base_census_variancia)
    X_census_variancia = newLabelEncoder(X_census_variancia)
    X_census_variancia = scaler(X_census_variancia)
    X_census_variancia = encoder(X_census_variancia)
    X_census_treinamento_var, X_census_teste_var, y_census_treinamento_var, y_census_teste_var = split(
        X_census_variancia, y_census_variancia)
    random_forest_var = model(X_census_treinamento_var,
                              y_census_treinamento_var)
    metrics(random_forest_var, X_census_teste_var, y_census_teste_var)


main()
