
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    # axis 0, pois quero concatenar as linhas
    X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
    y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
    return X_credit, y_credit


def createModel(X_credit, y_credit):
    classificador_rede_neural = MLPClassifier(
        activation='relu', batch_size=56, solver='adam')
    classificador_rede_neural.fit(X_credit, y_credit)

    classificador_arvore = DecisionTreeClassifier(
        criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    classificador_arvore.fit(X_credit, y_credit)

    classificador_svm = SVC(C=2.0, kernel='rbf', probability=True)
    classificador_svm.fit(X_credit, y_credit)

    return classificador_rede_neural, classificador_arvore, classificador_svm


def saveModel(classificador_rede_neural, classificador_arvore, classificador_svm):
    pickle.dump(classificador_rede_neural, open(
        'rede_neural_finalizado.sav', 'wb'))
    pickle.dump(classificador_arvore, open('arvore_finalizado.sav', 'wb'))
    pickle.dump(classificador_svm, open('svm_finalizado.sav', 'wb'))


def main():
    X_credit, y_credit = getData()
    classificador_rede_neural, classificador_arvore, classificador_svm = createModel(
        X_credit, y_credit)
    saveModel(classificador_rede_neural,
              classificador_arvore, classificador_svm)


main()
