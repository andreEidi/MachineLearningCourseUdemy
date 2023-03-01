import pandas as pd
from apyori import apriori


def getData():
    base_mercado1 = pd.read_csv(
        '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/mercado.csv', header=None)
    return base_mercado1


def transctions(base_mercado1):
    transacoes = []
    for i in range(len(base_mercado1)):
        # print(i)
        # print(base_mercado1.values[i, 0])
        transacoes.append([str(base_mercado1.values[i, j])
                           for j in range(base_mercado1.shape[1])])
        print("transacoes: ", transacoes[i])
    return transacoes


def rules(transacoes):
    regras = apriori(transacoes, min_support=0.3,
                     min_confidence=0.8, min_lift=2)
    resultados = list(regras)
    print(f"Há {len(resultados)} regras")
    for i in range(len(resultados)):
        print(f"Resultados {i}: ", resultados[i])
    return resultados


def info(resultados):
    A = []
    B = []
    suporte = []
    confianca = []
    lift = []

    for resultado in resultados:
        s = resultado[1]
        result_rules = resultado[2]
        for result_rule in result_rules:
            a = list(result_rule[0])
            b = list(result_rule[1])
            c = result_rule[2]
            l = result_rule[3]
            A.append(a)
            B.append(b)
            suporte.append(s)
            confianca.append(c)
            lift.append(l)
    for i in range(len(A)):
        print(
            f"SE {A[i]} ENTÃO {B[i]}. Com confiança {confianca[i]} e lift {lift[i]}")
    print("suporte: ", suporte)
    return A, B, suporte, confianca, lift


def toDataframe(A, B, suporte, confianca, lift):
    rules_df = pd.DataFrame(
        {'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})
    print(rules_df.sort_values(by='lift', ascending=False))


def main():
    base_mercado1 = getData()
    transacoes = transctions(base_mercado1)
    resultados = rules(transacoes)
    A, B, suporte, confianca, lift = info(resultados)
    toDataframe(A, B, suporte, confianca, lift)


main()
