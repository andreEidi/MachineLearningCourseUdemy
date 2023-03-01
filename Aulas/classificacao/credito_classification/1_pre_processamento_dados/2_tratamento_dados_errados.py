# TRATAMENTO DE VALORES INCONSISTENTES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

base_credit = pd.read_csv(
    '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/credit_data.csv')

# Localização de registros
print(base_credit.loc[base_credit['age'] < 0], end='\n\n')

# Apagar uma coluna inteira
# axis 1 para colunas
base_credit2 = base_credit.drop('age', axis=1)
print(base_credit2.head(), end='\n\n')

# apagar somente registros com valores inconsistentes
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit3.loc[base_credit['age'] < 0], '\n\n')

# Opção, preencher manualmente os dados errados com, por exemplo, a média
# retorna a média de todos os atributos
base_credit.mean()
# retorna a média das idades contando os valores inconsistentes
base_credit['age'].mean()
mediaIdades = base_credit['age'][base_credit['age'] > 0].mean()
# atribui a média das idades para os valores errados, indicando a coluna a ser mudada
base_credit.loc[base_credit['age'] < 0, 'age'] = mediaIdades
print(base_credit.loc[base_credit['age'] < 0], end='\n\n')
print(base_credit.head(27))
