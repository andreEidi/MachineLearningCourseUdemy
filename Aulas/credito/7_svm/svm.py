import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def algorithm(X_credit_treinamento, y_credit_treinamento):
    svm_credit = SVC(random_state=1, kernel='rbf', C=2)
    # O parametro c indica o quão 'perfeita' será a divisão da base
    svm_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)
    return svm_credit


def printInfo(svm_credit, X_credit_teste, y_credit_teste):
    previsoes = svm_credit.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))


def confusionMatrixPrint(svm_credit, X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste):
    cm = ConfusionMatrix(svm_credit)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.show()


def main():
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = getData()
    svm_credit = algorithm(X_credit_treinamento, y_credit_treinamento)
    printInfo(svm_credit, X_credit_teste, y_credit_teste)
    confusionMatrixPrint(svm_credit, X_credit_treinamento,
                         y_credit_treinamento, X_credit_teste, y_credit_teste)


main()
