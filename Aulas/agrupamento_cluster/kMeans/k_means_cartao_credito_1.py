import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
# Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


def data():
    base_cartao = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/credit_card_clients.csv', header=1)
    return base_cartao


def addCollum(base_cartao):
    # total da fatura nos seis meses
    base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + \
        base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + \
        base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
    print(base_cartao.head())
    return base_cartao


def normalizeScale(base_cartao):
    X_cartao = base_cartao.iloc[:, [1, 25]].values
    print("X_cartao: \n", X_cartao)
    # precisamos escalonar os dados
    scaler_cartao = StandardScaler()
    X_cartao = scaler_cartao.fit_transform(X_cartao)
    return X_cartao


def findK(X_cartao):
    wcss = []
    for i in range(1, 11):
        kmeans_cartao = KMeans(n_clusters=i, random_state=0)
        kmeans_cartao.fit(X_cartao)
        wcss.append(kmeans_cartao.inertia_)
    # Com base nos resultados para varios valores de k vamos decidir o melhor valor do mesmo
    grafico = px.line(x=range(1, 11), y=wcss)
    grafico.show()


def model(X_cartao):
    # kmeans_salario = KMeans(n_clusters=3)
    # kmeans_salario.fit(base_cartao)
    kmeans_cartao = KMeans(n_clusters=4, random_state=0)
    # faz o treinamento e já tem os rotulos
    rotulos = kmeans_cartao.fit_predict(X_cartao)
    return rotulos


def newGraphs(X_cartao, rotulos):
    grafico = px.scatter(x=X_cartao[:, 0], y=X_cartao[:, 1], color=rotulos)
    grafico.show()


def info(base_cartao, rotulos):
    # todos os clientes com sua classificação na ultima coluna
    lista_clientes = np.column_stack((base_cartao, rotulos))
    print("lista_clientes:\n", lista_clientes)
    lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
    print("lista_clientes ordenada:\n", lista_clientes)


def main():
    base_cartao = data()
    base_cartao = addCollum(base_cartao)
    X_cartao = normalizeScale(base_cartao)
    findK(X_cartao)
    rotulos = model(X_cartao)
    newGraphs(X_cartao, rotulos)
    info(base_cartao, rotulos)


main()
