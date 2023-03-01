import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
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
    print("Colunas: ", base_cartao.columns)
    return base_cartao


def normalizeScale(base_cartao):
    X_cartao_mais = base_cartao.iloc[:, [1, 2, 3, 4, 5, 25]].values
    print("X_cartao_mais: \n", X_cartao_mais)
    # precisamos escalonar os dados
    scaler_cartao_mais = StandardScaler()
    X_cartao_mais = scaler_cartao_mais.fit_transform(X_cartao_mais)
    return X_cartao_mais


def findK(X_cartao_mais):
    wcss = []
    for i in range(1, 11):
        kmeans_cartao_mais = KMeans(n_clusters=i, random_state=0)
        kmeans_cartao_mais.fit(X_cartao_mais)
        wcss.append(kmeans_cartao_mais.inertia_)
    # Com base nos resultados para varios valores de k vamos decidir o melhor valor do mesmo
    grafico = px.line(x=range(1, 11), y=wcss)
    grafico.show()


def model(X_cartao_mais):
    kmeans_cartao_mais = KMeans(n_clusters=2, random_state=0)
    # faz o treinamento e já tem os rotulos
    rotulos = kmeans_cartao_mais.fit_predict(X_cartao_mais)
    print("Rotulos: ", rotulos)
    return rotulos


def PCA_Function(X_cartao_mais):
    # reduzir a dimensialidade
    # Note que temos 6 atributos, a função retorna somente dois atributos relacionando das 6 aquelas que são parecidas e podem ser relacionadas
    pca = PCA(n_components=2)
    X_cartao_mais_pca = pca.fit_transform(X_cartao_mais)
    print("Shape: ", X_cartao_mais_pca.shape)
    print("X_cartao_mais_pca: \n", X_cartao_mais_pca)

    return X_cartao_mais_pca


def newGraphs(X_cartao_mais_pca, rotulos):
    grafico = px.scatter(
        x=X_cartao_mais_pca[:, 0], y=X_cartao_mais_pca[:, 1], color=rotulos)
    grafico.show()


def info(base_cartao, rotulos):
    # todos os clientes com sua classificação na ultima coluna
    lista_clientes = np.column_stack((base_cartao, rotulos))
    lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
    print("lista_clientes ordenada:\n", lista_clientes)


def main():
    base_cartao = data()
    base_cartao = addCollum(base_cartao)
    X_cartao_mais = normalizeScale(base_cartao)
    findK(X_cartao_mais)
    rotulos = model(X_cartao_mais)
    X_cartao_mais_pca = PCA_Function(X_cartao_mais)
    newGraphs(X_cartao_mais_pca, rotulos)
    info(base_cartao, rotulos)


main()
