import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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
    # print(base_cartao.head())
    return base_cartao


def normalizeScale(base_cartao):
    X_cartao = base_cartao.iloc[:, [1, 25]].values
    # print("X_cartao: \n", X_cartao)
    # precisamos escalonar os dados
    scaler_cartao = StandardScaler()
    X_cartao = scaler_cartao.fit_transform(X_cartao)
    return X_cartao


def dendrograma(X_cartao):
    dendrograma = dendrogram(linkage(X_cartao, method='ward'))
    plt.show()


def model(X_cartao):
    hc_cartao = AgglomerativeClustering(
        n_clusters=3, affinity='euclidean', linkage='ward')
    rotulos = hc_cartao.fit_predict(X_cartao)
    print("rotulos: ", rotulos)
    return rotulos


def newGraphs(X_cartao, rotulos):
    grafico = px.scatter(x=X_cartao[:, 0], y=X_cartao[:, 1], color=rotulos)
    grafico.show()


def main():
    base_cartao = data()
    base_cartao = addCollum(base_cartao)
    X_cartao = normalizeScale(base_cartao)
    dendrograma(X_cartao)
    # rotulos = model(X_cartao)
    # newGraphs(X_cartao, rotulos)


# Rodar amanhã
main()
