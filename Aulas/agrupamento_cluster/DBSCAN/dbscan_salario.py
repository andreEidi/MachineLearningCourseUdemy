import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


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


def model(base_salario):
    dbscan_salario = DBSCAN(eps=0.95, min_samples=2)
    # eps = alcance para a seleção da vizinhança
    # min_sample deve se ter n pontos para considerar como um cluster
    dbscan_salario.fit(base_salario)
    return dbscan_salario


def info(dbscan_salario):
    rotulos = dbscan_salario.labels_
    print("Rotulos: ", rotulos)
    return rotulos


def newGraphs(base_salario, rotulos):

    grafico = px.scatter(
        x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
    grafico.show()


def main():
    x, y = data()
    base_salario = buildData()
    base_salario, scaler_salario = normalizeScale(base_salario)

    dbscan_salario = model(base_salario)
    rotulos = info(dbscan_salario)
    newGraphs(base_salario, rotulos)


main()
