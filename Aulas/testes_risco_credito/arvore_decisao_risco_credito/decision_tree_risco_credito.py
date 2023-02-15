import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def readData():
    with open('/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/testes_risco_credito/risco_credito.pkl', 'rb') as f:
        X_risco_credito, y_risco_credito = pickle.load(f)
    return X_risco_credito, y_risco_credito


def algorithm(X_risco_credito, y_risco_credito):
    arvore_risco = DecisionTreeClassifier(criterion='entropy')
    arvore_risco.fit(X=X_risco_credito, y=y_risco_credito)
    return arvore_risco


def printInfo(arvore_risco):
    # imprime a ordem de importância de cada classe
    print(arvore_risco.feature_importances_)
    # Da nome as features
    previsores = ['historia', 'divida', 'garantias', 'renda']
    # Configura o tamanho da imagem
    figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # Imprime a árvore de decisão gerada
    tree.plot_tree(arvore_risco, feature_names=previsores,
                   class_names=arvore_risco.classes_, filled=True)
    plt.show()
    previsoes = arvore_risco.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
    print(previsoes)


def main():
    X_risco_credito, y_risco_credito = readData()
    arvore_risco = algorithm(X_risco_credito, y_risco_credito)
    printInfo(arvore_risco)


main()
