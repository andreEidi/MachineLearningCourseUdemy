# import das bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

# exploração inicial dos dados
base_census = pd.read_csv(
    '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/census.csv')
print(base_census.describe(), end='\n\n')
print(base_census.head(), end='\n\n')
# print(base_census.isnull().sum())
# Note que não há valores nulos


# Visualização dos dados

print(np.unique(base_census['income'], return_counts=True))
# Com o unique verificamos os valores unicos da base para o atributo income.
# Além disso, com o valor de return_count = true obtemos a informação sobre a quantidade de cada valor
plt.ioff()

sns.countplot(x=base_census['income'])
plt.show()
# Note que pelo resultado da imagem a classe está desbalanceada

plt.hist(x=base_census['age'])
plt.show()
# Note que pelo histograma a grande maioria das idades se concentra entre 20 e 40 anos

plt.hist(x=base_census['education-num'])
plt.show()
# Note que pelo histograma temos um padrão de 10 anos de estudos para grande parte dos dados

plt.hist(x=base_census['hour-per-week'])
plt.show()
# Note que grande maioria trabalha em torno de 40 horas por semana

# Gerar gráficos dinâmicos
grafico = px.treemap(base_census, path=['workclass', 'age'])
grafico.show()
# Note que este tipo de gráfico nos ajuda a ter uma melhor visualização de como é a relação dos dados

grafico2 = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
grafico2.show()

grafico3 = px.parallel_categories(
    base_census, dimensions=['occupation', 'relationship'])
grafico3.show()
