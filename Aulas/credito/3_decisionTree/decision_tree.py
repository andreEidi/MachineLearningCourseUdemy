import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report


def getData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(
            f)
    return X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste


def algorithm(X_credit_treinamento, y_credit_treinamento):
    arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)
    return arvore_credit


def printInfo(arvore_credit, X_credit_teste, y_credit_teste):
    previsoes = arvore_credit.predict(X_credit_teste)
    print(accuracy_score(y_credit_teste, previsoes))
    print(classification_report(y_credit_teste, previsoes))

    # imprime a ordem de importância de cada classe
    print(arvore_credit.feature_importances_)
    # Da nome as features
    previsores = ['income', 'age', 'loan']
    # Configura o tamanho da imagem
    figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # Imprime a árvore de decisão gerada
    tree.plot_tree(arvore_credit, feature_names=previsores,
                   class_names=['0', '1'], filled=True)
    plt.show()


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
