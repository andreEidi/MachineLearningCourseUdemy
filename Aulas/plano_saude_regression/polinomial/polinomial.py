import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def getData():
    base_plano_saude2 = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/plano_saude2.csv')
    X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
    y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

    return X_plano_saude2, y_plano_saude2


def algorithm(X_plano_saude2):
    # grau que iremos elevar cada dado. Teremos para cada dados os n valores. ex: x⁰, x¹, x²....x^n
    poly = PolynomialFeatures(degree=4)
    X_plano_saude2_poly = poly.fit_transform(X_plano_saude2)
    return X_plano_saude2_poly


def model(X_plano_saude2_poly, y_plano_saude2):
    regressor_saude_polinomial = LinearRegression()
    regressor_saude_polinomial.fit(X_plano_saude2_poly, y_plano_saude2)
    return regressor_saude_polinomial


def printInfo(regressor_saude_polinomial, X_plano_saude2_poly):
    # b0
    print(regressor_saude_polinomial.intercept_)
    # b1
    print(regressor_saude_polinomial.coef_)

    novo = [[40]]
    novo = PolynomialFeatures(degree=4).fit_transform(novo)
    print(regressor_saude_polinomial.predict(novo))

    previsoes = regressor_saude_polinomial.predict(X_plano_saude2_poly)
    print(previsoes)
    return previsoes


def scatterPlot(X_plano_saude2, y_plano_saude2, previsoes):
    grafico = px.scatter(x=X_plano_saude2[:, 0], y=y_plano_saude2)
    grafico.add_scatter(x=X_plano_saude2[:, 0],
                        y=previsoes, name='Regressão')
    grafico.show()


def main():
    X_plano_saude2, y_plano_saude2 = getData()
    X_plano_saude2_poly = algorithm(X_plano_saude2)
    regressor_saude_polinomial = model(X_plano_saude2_poly, y_plano_saude2)
    previsoes = printInfo(regressor_saude_polinomial, X_plano_saude2_poly)
    scatterPlot(X_plano_saude2, y_plano_saude2, previsoes)


main()
