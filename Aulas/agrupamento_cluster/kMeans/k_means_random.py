from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def data():
    X_random, y_random = make_blobs(n_samples=200, centers=5, random_state=1)
    return X_random, y_random


def graph(X_random, y_random):
    grafico = px.scatter(x=X_random[:, 0], y=X_random[:, 1])
    grafico.show()


def model(X_random):
    kmeans_blobs = KMeans(n_clusters=5)
    kmeans_blobs.fit(X_random)
    return kmeans_blobs


def info(kmeans_blobs, X_random):
    rotulos = kmeans_blobs.predict(X_random)
    print("rotulos: ", rotulos)

    centroides = kmeans_blobs.cluster_centers_
    print("centroides \n", centroides)

    return centroides, rotulos


def newGraphs(X_random, rotulos, centroides):

    grafico1 = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=rotulos)
    grafico2 = px.scatter(x=centroides[:, 0],
                          y=centroides[:, 1], size=[5, 5, 5, 5, 5])
    grafico3 = go.Figure(data=grafico1.data + grafico2.data)
    grafico3.show()


def main():
    x, y = data()
    graph(x, y)
    kmeans_blobs = model(x)
    centroides, rotulos = info(kmeans_blobs, x)
    newGraphs(x, rotulos, centroides)


main()
