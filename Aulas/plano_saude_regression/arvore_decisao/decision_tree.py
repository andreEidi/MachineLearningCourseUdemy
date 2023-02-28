import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


def getData():
    base_plano_saude2 = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/plano_saude2.csv')
    X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
    y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

    return X_plano_saude2, y_plano_saude2


# def algorithm(X_plano_saude2):
#     # grau que iremos elevar cada dado. Teremos para cada dados os n valores. ex: x⁰, x¹, x²....x^n
#     poly = PolynomialFeatures(degree=4)
#     X_plano_saude2 = poly.fit_transform(X_plano_saude2)
#     return X_plano_saude2


def model(X_plano_saude2, y_plano_saude2):
    regressor_saude_arvore = DecisionTreeRegressor()
    regressor_saude_arvore.fit(X_plano_saude2, y_plano_saude2)
    return regressor_saude_arvore


def printInfo(regressor_saude_arvore, X_plano_saude2, y_plano_saude2):

    print("score: ", regressor_saude_arvore.score(
        X_plano_saude2, y_plano_saude2))

    previsoes = regressor_saude_arvore.predict(X_plano_saude2)
    print(previsoes)
    return previsoes


def scatterPlot(X_plano_saude2, y_plano_saude2, previsoes):
    grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
    grafico.add_scatter(x=X_plano_saude2.ravel(),
                        y=previsoes, name='Regressão')
    grafico.show()


def SplitsEvidentes(regressor_saude_arvore, X_plano_saude2, y_plano_saude2, previsoes):
    X_teste_arvore = np.arange(min(X_plano_saude2), max(X_plano_saude2), 0.1)
    # transforma-lo em matriz
    X_teste_arvore = X_teste_arvore.reshape(-1, 1)
    grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
    grafico.add_scatter(x=X_teste_arvore.ravel(), y=regressor_saude_arvore.predict(
        X_teste_arvore), name='Regressão')
    grafico.show()


def main():
    X_plano_saude2, y_plano_saude2 = getData()
    # X_plano_saude2 = algorithm(X_plano_saude2)
    regressor_saude_arvore = model(X_plano_saude2, y_plano_saude2)
    previsoes = printInfo(regressor_saude_arvore,
                          X_plano_saude2, y_plano_saude2)
    scatterPlot(X_plano_saude2, y_plano_saude2, previsoes)
    SplitsEvidentes(regressor_saude_arvore, X_plano_saude2,
                    y_plano_saude2, previsoes)


main()
