import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def algorithm(X_credit_treinamento, y_credit_treinamento):
    # n_estimators = número de árvores
    arvore_credit = RandomForestClassifier(
        n_estimators=40, criterion='entropy', random_state=0)
    arvore_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)
    return arvore_credit


def printInfo(arvore_credit, X_credit_teste, y_credit_teste):
    previsoes = arvore_credit.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))


def confusionMatrixPrint(arvore_credit, X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste):
    cm = ConfusionMatrix(arvore_credit)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.show()


def main():
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = getData()
    arvore_credit = algorithm(X_credit_treinamento, y_credit_treinamento)
    printInfo(arvore_credit, X_credit_teste, y_credit_teste)
    confusionMatrixPrint(arvore_credit, X_credit_treinamento,
                         y_credit_treinamento, X_credit_teste, y_credit_teste)


main()
