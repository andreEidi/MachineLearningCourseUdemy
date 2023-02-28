from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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


def gridSearchDecisionTree(X_credit, y_credit):
    parametros = {'criterion': ['gini', 'entropy'], 'splitter': [
        'best', 'random'], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)

    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_

    print(melhores_parametros)
    print(melhor_resultado)


def gridSearchRandomForest(X_credit, y_credit):
    parametros = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [10, 40, 100, 150],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 5, 10]}
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)


def gridSearchKNN(X_credit, y_credit):
    parametros = {'n_neighbors': [3, 5, 10, 20],
                  'p': [1, 2]}
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)


def gridSearchLogisticRegression(X_credit, y_credit):
    parametros = {'tol': [0.0001, 0.00001, 0.000001],
                  'C': [1.0, 1.5, 2.0],
                  'solver': ['lbfgs', 'sag', 'saga']}
    grid_search = GridSearchCV(
        estimator=LogisticRegression(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)


def gridSearchSVM(X_credit, y_credit):
    parametros = {'tol': [0.001, 0.0001, 0.00001],
                  'C': [1.0, 1.5, 2.0],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)


def gridSearchNeural(X_credit, y_credit):
    parametros = {'activation': ['relu', 'logistic', 'tahn'],
                  'solver': ['adam', 'sgd'],
                  'batch_size': [10, 56]}
    grid_search = GridSearchCV(
        estimator=MLPClassifier(), param_grid=parametros)
    grid_search.fit(X_credit, y_credit)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    print(melhores_parametros)
    print(melhor_resultado)


def main():
    X_credit, y_credit = getData()
    gridSearchDecisionTree(X_credit, y_credit)
    gridSearchRandomForest(X_credit, y_credit)
    gridSearchKNN(X_credit, y_credit)
    gridSearchLogisticRegression(X_credit, y_credit)
    gridSearchSVM(X_credit, y_credit)
    gridSearchNeural(X_credit, y_credit)


main()
