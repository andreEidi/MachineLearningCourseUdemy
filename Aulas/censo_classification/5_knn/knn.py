import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def algorithm(X_census_treinamento, y_census_treinamento):
    knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    # minkowski = como a distancia é calculada, com p = 2 é euclidiana
    knn_credit.fit(X=X_census_treinamento, y=y_census_treinamento)
    return knn_credit


def printInfo(knn_credit, X_census_teste, y_census_teste):
    previsoes = knn_credit.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(knn_credit, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(knn_credit)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    knn_credit = algorithm(X_census_treinamento, y_census_treinamento)
    printInfo(knn_credit, X_census_teste, y_census_teste)
    confusionMatrixPrint(knn_credit, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
