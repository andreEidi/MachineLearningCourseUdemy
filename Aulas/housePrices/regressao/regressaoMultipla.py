import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# def viewData(base_casas):
#     print(base_casas.describe())
#     base_casas.isnull().sum()  # Não há nenhuma celula nula
#     print(base_casas.corr())  # Verifica a correlação entre as classes
#     figura = plt.figure(figsize=(20, 20))
#     # plt.ioff()
#     sns.heatmap(base_casas.corr(numeric_only=True),
#                 annot=True)  # annot coloca os valores
#     plt.show()


def getData():
    base_casas = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/house_prices.csv')
    # viewData(base_casas)

    X_casas = base_casas.iloc[:, 3:19].values
    y_casas = base_casas.iloc[:, 2].values

    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
        X_casas, y_casas, test_size=0.3, random_state=0)

    return X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste


def algorithm(X_casas_treinamento, y_casas_treinamento):
    regressor_multiplo_casas = LinearRegression()
    regressor_multiplo_casas.fit(X_casas_treinamento, y_casas_treinamento)
    return regressor_multiplo_casas


def printInfo(regressor_multiplo_casas, X_casas_treinamento, y_casas_treinamento, X_casas_teste, y_casas_teste):
    # b0
    print(regressor_multiplo_casas.intercept_)
    # b1
    print(regressor_multiplo_casas.coef_)
    print(len(regressor_multiplo_casas.coef_))

    print(regressor_multiplo_casas.score(
        X_casas_treinamento, y_casas_treinamento))
    print(regressor_multiplo_casas.score(X_casas_teste, y_casas_teste))

    previsoes = regressor_multiplo_casas.predict(X_casas_teste)
    return previsoes


def dadosTeste_erro(X_casas_teste, y_casas_teste, regressor_multiplo_casas, previsoes):
    print(mean_absolute_error(y_casas_teste, previsoes))


def main():
    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = getData()
    regressor_multiplo_casas = algorithm(
        X_casas_treinamento, y_casas_treinamento)
    previsoes = printInfo(regressor_multiplo_casas, X_casas_treinamento,
                          y_casas_treinamento, X_casas_teste, y_casas_teste)
    dadosTeste_erro(X_casas_teste, y_casas_teste,
                    regressor_multiplo_casas, previsoes)


main()
