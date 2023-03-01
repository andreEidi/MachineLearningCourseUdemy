import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


def data():
    x = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]
    y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100,
         3000, 5900, 4100, 5100, 7000, 5000, 6500]
    return x, y


def buildData():
    base_salario = np.array([[20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900],
                             [53, 950], [55, 2000], [47, 2100], [
                                 52, 3000], [32, 5900],
                             [39, 4100], [41, 5100], [39, 7000], [48, 5000], [48, 6500]])
    return base_salario


def normalizeScale(base_salario):
    scaler_salario = StandardScaler()
    base_salario = scaler_salario.fit_transform(base_salario)
    return base_salario, scaler_salario


def graph(base_salario):
    grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1])
    grafico.show()


def dendrograma(base_salario):
    dendrograma = dendrogram(linkage(base_salario, method='ward'))
    plt.title('Dendrograma')
    plt.xlabel('Pessoas')
    plt.ylabel('Distância')
    plt.show()
    # Note que no caso o número ideal de clusters é 3


def model(base_salario):
    hc_salario = AgglomerativeClustering(
        n_clusters=3, linkage='ward', affinity='euclidean')
    rotulos = hc_salario.fit_predict(base_salario)
    print("rotulos: ", rotulos)
    return rotulos


def newGraphs(base_salario, rotulos):
    grafico = px.scatter(
        x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
    grafico.show()


def main():
    x, y = data()
    base_salario = buildData()
    base_salario, scaler_salario = normalizeScale(base_salario)
    graph(base_salario)
    dendrograma(base_salario)
    rotulos = model(base_salario)
    newGraphs(base_salario, rotulos)


main()
