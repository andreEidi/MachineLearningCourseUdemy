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
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def model(X_census_treinamento, y_census_treinamento):
    naive_credit_data = GaussianNB()
    naive_credit_data.fit(X_census_treinamento, y_census_treinamento)
    return naive_credit_data


def printInfo(naive_credit_data, X_census_teste, y_census_teste):
    previsoes = naive_credit_data.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(confusion_matrix(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(naive_credit_data, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(naive_credit_data)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    naive_credit_data = model(X_census_treinamento, y_census_treinamento)
    printInfo(naive_credit_data, X_census_teste, y_census_teste)
    confusionMatrixPrint(naive_credit_data, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
