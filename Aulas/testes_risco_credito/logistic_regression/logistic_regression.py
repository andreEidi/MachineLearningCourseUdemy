import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def readData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/testes_risco_credito/risco_credito.pkl', 'rb') as f:
        X_risco_credito, y_risco_credito = pickle.load(f)

    # Axis = 0 indica que quero apagar linhas
    X_risco_credito = np.delete(X_risco_credito, [2, 7, 11], axis=0)
    y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)
    return X_risco_credito, y_risco_credito


def algorithm(X_risco_credito, y_risco_credito):
    logistic_risco_credito = LogisticRegression(random_state=1)
    logistic_risco_credito.fit(X=X_risco_credito, y=y_risco_credito)
    return logistic_risco_credito


def printInfo(logistic_risco_credito):
    print(logistic_risco_credito.intercept_)
    print(logistic_risco_credito.coef_)

    previsoes = logistic_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
    print(previsoes)


def main():
    X_risco_credito, y_risco_credito = readData()
    logistic_risco_credito = algorithm(X_risco_credito, y_risco_credito)
    printInfo(logistic_risco_credito)


main()
