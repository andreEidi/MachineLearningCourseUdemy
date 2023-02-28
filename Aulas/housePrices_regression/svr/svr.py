import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def getData():
    base_casas = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/house_prices.csv')

    X_casas = base_casas.iloc[:, 3:19].values
    y_casas = base_casas.iloc[:, 2].values

    X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
        X_casas, y_casas, test_size=0.3, random_state=0)

    scaler_x_casas = StandardScaler()
    X_casas_treinamento_scaled = scaler_x_casas.fit_transform(
        X_casas_treinamento)
    scaler_y_casas = StandardScaler()
    y_casas_treinamento_scaled = scaler_y_casas.fit_transform(
        y_casas_treinamento.reshape(-1, 1))
    X_casas_teste_scaled = scaler_x_casas.transform(X_casas_teste)
    y_casas_teste_scaled = scaler_y_casas.transform(
        y_casas_teste.reshape(-1, 1))

    return X_casas_treinamento_scaled, X_casas_teste_scaled, y_casas_treinamento_scaled, y_casas_teste_scaled, scaler_y_casas


def model(X_casas_treinamento_scaled, y_casas_treinamento_scaled):
    regressor_svr_casas = SVR(kernel='rbf')
    regressor_svr_casas.fit(X_casas_treinamento_scaled,
                            y_casas_treinamento_scaled.ravel())
    return regressor_svr_casas


def printInfo(regressor_svr_casas, X_casas_treinamento_scaled, y_casas_treinamento_scaled, X_casas_teste_scaled, y_casas_teste_scaled, scaler_y_casas):

    print("score: ", regressor_svr_casas.score(
        X_casas_treinamento_scaled, y_casas_treinamento_scaled))

    print(regressor_svr_casas.score(X_casas_teste_scaled, y_casas_teste_scaled))

    previsoes = regressor_svr_casas.predict(
        X_casas_teste_scaled).reshape(-1, 1)
    print(previsoes)

    # retornar aos valores não escalonados
    y_casas_teste_inverse = scaler_y_casas.inverse_transform(
        y_casas_teste_scaled)
    previsoes_inverse = scaler_y_casas.inverse_transform(previsoes)
    print("y_casas_teste_inverse: ", y_casas_teste_inverse)
    print("previsoes_inverse: ", previsoes_inverse)

    return previsoes


def main():
    X_casas_treinamento_scaled, X_casas_teste_scaled, y_casas_treinamento_scaled, y_casas_teste_scaled, scaler_y_casas = getData()
    regressor_svr_casas = model(
        X_casas_treinamento_scaled, y_casas_treinamento_scaled)
    previsoes = printInfo(regressor_svr_casas,
                          X_casas_treinamento_scaled, y_casas_treinamento_scaled, X_casas_teste_scaled, y_casas_teste_scaled, scaler_y_casas)


main()
