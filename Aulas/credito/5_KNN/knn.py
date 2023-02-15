import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def algorithm(X_credit_treinamento, y_credit_treinamento):
    knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    # minkowski = como a distancia é calculada, com p = 2 é euclidiana
    knn_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)
    return knn_credit


def printInfo(knn_credit, X_credit_teste, y_credit_teste):
    previsoes = knn_credit.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))


def confusionMatrixPrint(knn_credit, X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste):
    cm = ConfusionMatrix(knn_credit)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.show()


def main():
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = getData()
    knn_credit = algorithm(X_credit_treinamento, y_credit_treinamento)
    printInfo(knn_credit, X_credit_teste, y_credit_teste)
    confusionMatrixPrint(knn_credit, X_credit_treinamento,
                         y_credit_treinamento, X_credit_teste, y_credit_teste)


main()
