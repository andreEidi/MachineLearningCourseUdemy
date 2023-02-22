import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neural_network import MLPClassifier  # multilayer perceptor


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def algorithm(X_credit_treinamento, y_credit_treinamento):
    rede_neural_credit = MLPClassifier(
        max_iter=1500, verbose=True, tol=0.0000100, solver='adam', activation='relu', hidden_layer_sizes=(2, 2))
    # verbose mostra o erro para cada época, max_iter indica o maximo de épocas
    # tol  indica a tolerancia/erro maximo de uma epoca para outra
    # solver é o algoritmo que faz a otimização
    # activation é a função de ativação
    # hidden layer sizes, número de neuronios na camada oculta, no caso, temos 2 camadas de 2 neuronios cada
    rede_neural_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)
    return rede_neural_credit


def printInfo(rede_neural_credit, X_credit_teste, y_credit_teste):
    previsoes = rede_neural_credit.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))


def confusionMatrixPrint(rede_neural_credit, X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste):
    cm = ConfusionMatrix(rede_neural_credit)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.show()


def main():
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = getData()
    rede_neural_credit = algorithm(X_credit_treinamento, y_credit_treinamento)
    printInfo(rede_neural_credit, X_credit_teste, y_credit_teste)
    confusionMatrixPrint(rede_neural_credit, X_credit_treinamento,
                         y_credit_treinamento, X_credit_teste, y_credit_teste)


main()
