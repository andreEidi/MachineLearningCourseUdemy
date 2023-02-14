from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos
import pickle
from sklearn.naive_bayes import GaussianNB


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def model(X_credit_treinamento, y_credit_treinamento):
    naive_credit_data = GaussianNB()
    naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)
    return naive_credit_data


def printInfo(naive_credit_data, X_credit_teste, y_credit_teste):
    previsoes = naive_credit_data.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(confusion_matrix(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))


def confusionMatrixPrint(naive_credit_data, X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste):
    cm = ConfusionMatrix(naive_credit_data)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.show()


def main():
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = getData()
    naive_credit_data = model(X_credit_treinamento, y_credit_treinamento)
    printInfo(naive_credit_data, X_credit_teste, y_credit_teste)
    confusionMatrixPrint(naive_credit_data, X_credit_treinamento,
                         y_credit_treinamento, X_credit_teste, y_credit_teste)


main()
