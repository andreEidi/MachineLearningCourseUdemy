import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def algorithm(X_census_treinamento, y_census_treinamento):
    svm_census = SVC(random_state=1, kernel='linear', C=2)
    # O parametro c indica o quão 'perfeita' será a divisão da base
    svm_census.fit(X=X_census_treinamento, y=y_census_treinamento)
    return svm_census


def printInfo(svm_census, X_census_teste, y_census_teste):
    previsoes = svm_census.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(svm_census, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(svm_census)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    svm_census = algorithm(X_census_treinamento, y_census_treinamento)
    printInfo(svm_census, X_census_teste, y_census_teste)
    confusionMatrixPrint(svm_census, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
