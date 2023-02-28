import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


def getData():
    base_casas = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/house_prices.csv')
    # viewData(base_casas)

    X_casas = base_casas.iloc[:, 3:19].values
    y_casas = base_casas.iloc[:, 2].values

    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
        X_casas, y_casas, test_size=0.3, random_state=0)

    return X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste


def model(X_casas_treinamento, y_casas_treinamento):
    regressor_saude_arvore = DecisionTreeRegressor()
    regressor_saude_arvore.fit(X_casas_treinamento, y_casas_treinamento)
    return regressor_saude_arvore


def printInfo(regressor_saude_arvore, X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste):

    print("score_treinamento: ", regressor_saude_arvore.score(
        X_casas_treinamento, y_casas_treinamento))
    print("score_teste: ", regressor_saude_arvore.score(
        X_casas_teste, y_casas_teste))

    previsoes = regressor_saude_arvore.predict(X_casas_teste)
    print(previsoes)
    return previsoes


def error(y_casas_teste, previsoes):
    print("Error: ", mean_absolute_error(y_casas_teste, previsoes))


def main():
    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = getData()
    # X_plano_saude2 = algorithm(X_plano_saude2)
    regressor_saude_arvore = model(X_casas_treinamento, y_casas_treinamento)
    previsoes = printInfo(regressor_saude_arvore, X_casas_treinamento,
                          X_casas_teste, y_casas_treinamento, y_casas_teste)
    error(y_casas_teste, previsoes)


main()
