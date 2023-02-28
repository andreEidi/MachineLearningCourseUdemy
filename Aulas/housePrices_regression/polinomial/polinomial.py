import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def getData():
    base_casas = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/house_prices.csv')
    # viewData(base_casas)

    X_casas = base_casas.iloc[:, 3:19].values
    y_casas = base_casas.iloc[:, 2].values

    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
        X_casas, y_casas, test_size=0.3, random_state=0)

    return X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste


def algorithm(X_casas_treinamento, X_casas_teste):
    # grau que iremos elevar cada dado. Teremos para cada dados os n valores. ex: x⁰, x¹, x²....x^n
    poly = PolynomialFeatures(degree=2)
    X_casas_treinamento_poly = poly.fit_transform(X_casas_treinamento)
    X_casas_teste_poly = poly.fit_transform(X_casas_teste)
    return X_casas_treinamento_poly, X_casas_teste_poly


def model(X_casas_treinamento_poly, y_casas_treinamento):
    regressor_casas_polinomial = LinearRegression()
    regressor_casas_polinomial.fit(
        X_casas_treinamento_poly, y_casas_treinamento)
    return regressor_casas_polinomial


def printInfo(regressor_casas_polinomial, X_casas_treinamento_poly, X_casas_teste_poly, y_casas_treinamento, y_casas_teste):

    print("treinamento: ", regressor_casas_polinomial.score(
        X_casas_treinamento_poly, y_casas_treinamento))
    print("Teste: ", regressor_casas_polinomial.score(
        X_casas_teste_poly, y_casas_teste))

    previsoes = regressor_casas_polinomial.predict(X_casas_teste_poly)
    print(previsoes)
    return previsoes


def error(y_casas_teste, previsoes):
    print("Error: ", mean_absolute_error(y_casas_teste, previsoes))


def main():
    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = getData()
    X_casas_treinamento_poly, X_casas_teste_poly = algorithm(
        X_casas_treinamento, X_casas_teste)
    regressor_casas_polinomial = model(
        X_casas_treinamento_poly, y_casas_treinamento)
    previsoes = printInfo(regressor_casas_polinomial, X_casas_treinamento_poly,
                          X_casas_teste_poly, y_casas_treinamento, y_casas_teste)
    error(y_casas_teste, previsoes)


main()
