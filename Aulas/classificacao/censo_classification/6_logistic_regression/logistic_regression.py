import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/censo/census.pkl', 'rb') as f:
        X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(
            f)
    return X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste


def algorithm(X_census_treinamento, y_census_treinamento):
    logistic_census = LogisticRegression(random_state=1)
    logistic_census.fit(X=X_census_treinamento, y=y_census_treinamento)
    return logistic_census


def printInfo(logistic_census, X_census_teste, y_census_teste):
    print(logistic_census.intercept_)
    print(logistic_census.coef_)
    previsoes = logistic_census.predict(X_census_teste)
    print(accuracy_score(y_census_teste, previsoes))
    print(classification_report(y_census_teste, previsoes))


def confusionMatrixPrint(logistic_census, X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste):
    cm = ConfusionMatrix(logistic_census)
    cm.fit(X_census_treinamento, y_census_treinamento)
    cm.score(X_census_teste, y_census_teste)
    plt.show()


def main():
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = getData()
    logistic_census = algorithm(X_census_treinamento, y_census_treinamento)
    printInfo(logistic_census, X_census_teste, y_census_teste)
    confusionMatrixPrint(logistic_census, X_census_treinamento,
                         y_census_treinamento, X_census_teste, y_census_teste)


main()
