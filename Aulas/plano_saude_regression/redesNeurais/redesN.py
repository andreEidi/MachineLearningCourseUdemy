import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def getData():
    base_plano_saude2 = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/plano_saude2.csv')
    X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
    y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

    return X_plano_saude2, y_plano_saude2


def preProcessing(X_plano_saude2, y_plano_saude2):
    scaler_x = StandardScaler()
    X_plano_saude2_scaled = scaler_x.fit_transform(X_plano_saude2)
    scaler_y = StandardScaler()
    y_plano_saude2_scaled = scaler_y.fit_transform(
        y_plano_saude2.reshape(-1, 1))
    return X_plano_saude2_scaled, y_plano_saude2_scaled


def model(X_plano_saude2, y_plano_saude2):
    # note que os dados não estão normalizados. Isso (no SVM, faz diferença)
    regressor_saude_rna = MLPRegressor(max_iter=1000)
    regressor_saude_rna.fit(X_plano_saude2, y_plano_saude2.ravel())
    return regressor_saude_rna


def printInfo(regressor_saude_rna, X_plano_saude2, y_plano_saude2):

    print("score: ", regressor_saude_rna.score(
        X_plano_saude2, y_plano_saude2))

    previsoes = regressor_saude_rna.predict(X_plano_saude2)
    print(previsoes)
    return previsoes


def scatterPlot(regressor_saude_rna, X_plano_saude2_scaled, y_plano_saude2_scaled, previsoes):
    grafico = px.scatter(x=X_plano_saude2_scaled.ravel(),
                         y=y_plano_saude2_scaled.ravel())
    grafico.add_scatter(x=X_plano_saude2_scaled.ravel(
    ), y=regressor_saude_rna.predict(X_plano_saude2_scaled), name='Regressão')
    grafico.show()


def main():
    X_plano_saude2, y_plano_saude2 = getData()
    X_plano_saude2_scaled, y_plano_saude2_scaled = preProcessing(
        X_plano_saude2, y_plano_saude2)
    regressor_saude_rna = model(X_plano_saude2_scaled, y_plano_saude2_scaled)
    previsoes = printInfo(regressor_saude_rna,
                          X_plano_saude2_scaled, y_plano_saude2_scaled)
    scatterPlot(regressor_saude_rna, X_plano_saude2_scaled,
                y_plano_saude2_scaled, previsoes)


main()
