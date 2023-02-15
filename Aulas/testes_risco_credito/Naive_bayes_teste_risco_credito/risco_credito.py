import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB


def readCSV():
    return pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/risco_credito.csv')


def selectValues(base_risco_credito):
    X_risco_credito = base_risco_credito.iloc[:, 0:4].values
    y_risco_credito = base_risco_credito.iloc[:, 4].values
    return X_risco_credito, y_risco_credito


def mapLabels(X_risco_credito):
    label_encoder_historia = LabelEncoder()
    label_encoder_divida = LabelEncoder()
    label_encoder_garantia = LabelEncoder()
    label_encoder_renda = LabelEncoder()
    X_risco_credito[:, 0] = label_encoder_historia.fit_transform(
        X_risco_credito[:, 0])
    X_risco_credito[:, 1] = label_encoder_divida.fit_transform(
        X_risco_credito[:, 1])
    X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(
        X_risco_credito[:, 2])
    X_risco_credito[:, 3] = label_encoder_renda.fit_transform(
        X_risco_credito[:, 3])
    return X_risco_credito


def makeFilePreProcessing(X_risco_credito, y_risco_credito):
    with open('risco_credito.pkl', 'wb') as f:
        pickle.dump([X_risco_credito, y_risco_credito], f)


def algorithm(X_risco_credito, y_risco_credito):
    naive_risco_credito = GaussianNB()
    naive_risco_credito.fit(X=X_risco_credito, y=y_risco_credito)

    # historia boa (0), divida alta (0), garantias nenhuma (1), renda >35 (2)
    # historia ruim (2), divida alta (0), garantias adequada (0), renda <15 (0)
    previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
    print(previsao)
    return naive_risco_credito


def printInfo(naive_risco_credito):
    # imprimi as classes de saída do modelo
    print(naive_risco_credito.classes_)
    # contar quantos atributos temos por classe
    print(naive_risco_credito.class_count_)
    # ver como estão distribuidas as probabilidades destas classes a priori
    print(naive_risco_credito.class_prior_)


def main():
    base_risco_credito = readCSV()
    X_risco_credito, y_risco_credito = selectValues(base_risco_credito)
    X_risco_credito = mapLabels(X_risco_credito)
    # makeFilePreProcessing(X_risco_credito, y_risco_credito)
    naive_risco_credito = algorithm(X_risco_credito, y_risco_credito)
    printInfo(naive_risco_credito)


main()
