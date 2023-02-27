
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
    rede_neural = pickle.load(open(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/11_salvar_carregar_modelo/classificadores/rede_neural_finalizado.sav', 'rb'))
    arvore = pickle.load(open(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/11_salvar_carregar_modelo/classificadores/arvore_finalizado.sav', 'rb'))
    svm = pickle.load(open(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Aulas/credito/11_salvar_carregar_modelo/classificadores/svm_finalizado.sav', 'rb'))
    return rede_neural, arvore, svm, X_credit, y_credit


def createregister(X_credit):
    novo_registro = X_credit[1999]
    novo_registro = novo_registro.reshape(1, -1)
    return novo_registro


def predictions(novo_registro, rede_neural, arvore, svm):
    resposta_rede = rede_neural.predict(novo_registro)
    resposta_arv = arvore.predict(novo_registro)
    resposta_svm = svm.predict(novo_registro)
    print(resposta_rede[0], resposta_arv[0], resposta_svm[0])
    return resposta_rede, resposta_arv, resposta_svm


def results(resposta_rede, resposta_arv, resposta_svm):
    paga = 0
    nao_paga = 0

    if resposta_rede[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if resposta_arv[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if resposta_svm[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if paga > nao_paga:
        print('Cliente pagará o empréstimo')
    elif paga == nao_paga:
        print('Empate')
    else:
        print('Cliente não pagará o empréstimo')


def main():
    rede_neural, arvore, svm, X_credit, y_credit = getData()
    novo_registro = createregister(
        X_credit)
    resposta_rede, resposta_arv, resposta_svm = predictions(
        novo_registro, rede_neural, arvore, svm)
    results(resposta_rede, resposta_arv, resposta_svm)


main()
