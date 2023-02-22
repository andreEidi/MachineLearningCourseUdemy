import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neural_network import MLPClassifier


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def algorithm(X_census_treinamento, y_census_treinamento):
    neural_censsus = MLPClassifier(
        verbose=False, max_iter=1000, tol=0.000010, hidden_layer_sizes=(55, 55))
    neural_censsus.fit(X=X_census_treinamento, y=y_census_treinamento)
    return neural_censsus


def printInfo(neural_censsus, X_census_teste, y_census_teste):
    previsoes = neural_censsus.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(neural_censsus, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(neural_censsus)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    neural_censsus = algorithm(X_census_treinamento, y_census_treinamento)
    printInfo(neural_censsus, X_census_teste, y_census_teste)
    confusionMatrixPrint(neural_censsus, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
