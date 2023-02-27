import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def getData():
    base_plano_saude = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/plano_saude.csv')
    X_plano_saude = base_plano_saude.iloc[:, 0].values
    y_plano_saude = base_plano_saude.iloc[:, 1].values
    # analisa a correlação entre as variaveis
    np.corrcoef(X_plano_saude, y_plano_saude)

    X_plano_saude = X_plano_saude.reshape(-1, 1)
    return X_plano_saude, y_plano_saude


def algorithm(X_plano_saude, y_plano_saude):
    regressor_plano_saude = LinearRegression()
    regressor_plano_saude.fit(X_plano_saude, y_plano_saude)
    return regressor_plano_saude


def printInfo(regressor_plano_saude, X_plano_saude):
    # b0
    print(regressor_plano_saude.intercept_)
    # b1
    print(regressor_plano_saude.coef_)
    previsoes = regressor_plano_saude.predict(X_plano_saude)
    print(previsoes)
    return previsoes


def scatterPlot(X_plano_saude, y_plano_saude, previsoes):
    # grafico de dispersão
    # ravel retorna para um vetor simples
    grafico = px.scatter(x=X_plano_saude.ravel(), y=y_plano_saude)
    grafico.add_scatter(x=X_plano_saude.ravel(), y=previsoes, name='Regressão')
    grafico.show()


def predicao(regressor_plano_saude):
    print(regressor_plano_saude.predict([[40]]))


def score(regressor_plano_saude, X_plano_saude, y_plano_saude):
    print(regressor_plano_saude.score(X_plano_saude, y_plano_saude))


def residual(regressor_plano_saude, X_plano_saude, y_plano_saude):
    # residual plot
    # A distancia dos pontos ate a linha de regressao
    from yellowbrick.regressor import ResidualsPlot
    visualizador = ResidualsPlot(regressor_plano_saude)
    visualizador.fit(X_plano_saude, y_plano_saude)
    visualizador.poof()


def main():
    X_plano_saude, y_plano_saude = getData()
    regressor_plano_saude = algorithm(X_plano_saude, y_plano_saude)
    previsoes = printInfo(regressor_plano_saude, X_plano_saude)
    scatterPlot(X_plano_saude, y_plano_saude, previsoes)
    predicao(regressor_plano_saude)
    score(regressor_plano_saude, X_plano_saude, y_plano_saude)
    residual(regressor_plano_saude, X_plano_saude, y_plano_saude)


main()
