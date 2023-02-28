from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def algorithm(X_census_treinamento, y_census_treinamento):
    arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore_credit.fit(X=X_census_treinamento, y=y_census_treinamento)
    return arvore_credit


def printInfo(arvore_credit, X_census_teste, y_census_teste):
    previsoes = arvore_credit.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(arvore_credit, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(arvore_credit)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    arvore_credit = algorithm(X_census_treinamento, y_census_treinamento)
    printInfo(arvore_credit, X_census_teste, y_census_teste)
    confusionMatrixPrint(arvore_credit, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
