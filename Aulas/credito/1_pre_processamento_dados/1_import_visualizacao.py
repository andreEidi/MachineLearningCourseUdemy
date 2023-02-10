# import das bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

# importa a base de dados e visualização de dados
# 0 a pessoa pagou o empréstimo - 1 a pessoa não pagou
base_credit = pd.read_csv(
    '/home/andre/Projetos/Machine_learning_udemy_course/Bases_de_dados/credit_data.csv')
# imprime as 5 primeiras linhas da base de dados ou quantas quiser, basta colocar como parâmetro
print(base_credit.head(), end='\n\n')
# imprime as 5 últimas linhas da base
print(base_credit.tail(), end='\n\n')
# Imprime algumas informações sobre cada coluna da base de dados
print(base_credit.describe(), end='\n\n')
# filtro nos dados
print(base_credit[base_credit['income'] >= 69995.685578], end='\n\n')

# Visualização de dados
# conta os valores únicos que existem nessa coluna (default). Return_counts indica o número que cada valor aparece
print(np.unique(base_credit['default'], return_counts=True), end='\n\n')
# cria um gráfico com a quantidade que cada valor aparece
sns.countplot(x=base_credit['default'])
# disable interactive mode, assim posso imprimir duas figuras sem que os gráficos se sobreponham
plt.ioff()
# mostra a imagem
plt.show()
# monta um histograma
plt.hist(x=base_credit['age'])
plt.show()
# monta histograma para os dados de renda
plt.hist(x=base_credit['income'])
plt.show()
plt.hist(x=base_credit['loan'])
plt.show()
# Salva a imagem no diretório
# plt.savefig('my_plot.png')
# Gráfico dinêmico
# grafico de dispersão que relaciona atrinbutos
# color: seleciona um atributo para o qual vc quer observar melhor a relação
grafico = px.scatter_matrix(base_credit, dimensions=[
                            'age', 'income', 'loan'], color='default')
grafico.show()
