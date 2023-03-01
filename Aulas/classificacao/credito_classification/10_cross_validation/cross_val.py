from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd

# Realiza 300 testes. Isso com 10 conjuntos com 30 random state (distribuição dos dados)


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    # axis 0, pois quero concatenar as linhas
    X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
    y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
    return X_credit, y_credit


def crossValDecisionTree(X_credit, y_credit):
    resultados_arvore = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        arvore = DecisionTreeClassifier(
            criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
        scores = cross_val_score(arvore, X_credit, y_credit, cv=kfold)
        # print(scores)
        # print(scores.mean())
        resultados_arvore.append(scores.mean())
    return resultados_arvore


def crossValRandomForest(X_credit, y_credit):
    resultados_random_forest = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        random_forest = RandomForestClassifier(
            criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
        scores = cross_val_score(random_forest, X_credit, y_credit, cv=kfold)
        resultados_random_forest.append(scores.mean())
    return resultados_random_forest


def crossValKNN(X_credit, y_credit):
    resultados_knn = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        knn = KNeighborsClassifier()
        scores = cross_val_score(knn, X_credit, y_credit, cv=kfold)
        resultados_knn.append(scores.mean())
    return resultados_knn


def crossValLogisticRegression(X_credit, y_credit):
    resultados_logistica = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
        scores = cross_val_score(logistica, X_credit, y_credit, cv=kfold)
        resultados_logistica.append(scores.mean())
    return resultados_logistica


def crossValSVM(X_credit, y_credit):
    resultados_svm = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        svm = SVC(kernel='rbf', C=2.0)
        scores = cross_val_score(svm, X_credit, y_credit, cv=kfold)
        resultados_svm.append(scores.mean())
    return resultados_svm


def crossValNeural(X_credit, y_credit):
    resultados_rede_neural = []
    for i in range(30):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)

        rede_neural = MLPClassifier(
            activation='relu', batch_size=56, solver='adam')
        scores = cross_val_score(rede_neural, X_credit, y_credit, cv=kfold)
        resultados_rede_neural.append(scores.mean())
    return resultados_rede_neural


def main():
    X_credit, y_credit = getData()
    resultados_arvore = crossValDecisionTree(X_credit, y_credit)
    resultados_random_forest = crossValRandomForest(X_credit, y_credit)
    resultados_knn = crossValKNN(X_credit, y_credit)
    resultados_logistica = crossValLogisticRegression(X_credit, y_credit)
    resultados_svm = crossValSVM(X_credit, y_credit)
    resultados_rede_neural = crossValNeural(X_credit, y_credit)

    resultados = pd.DataFrame({'Arvore': resultados_arvore, 'Random forest': resultados_random_forest,
                               'KNN': resultados_knn, 'Logistica': resultados_logistica,
                               'SVM': resultados_svm, 'Rede neural': resultados_rede_neural})
    print(resultados)
    print(resultados.describe())


main()
