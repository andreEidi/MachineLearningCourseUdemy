# TRATAMENTO DE VALORES FALTANTES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

base_credit = pd.read_csv(
    '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/credit_data.csv')
mediaIdades = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = mediaIdades
print(base_credit.head())

# indica quais valores estão faltantes
base_credit.isnull()
# indica a quantidade de vezes que uma célula foi dada como nula
base_credit.isnull().sum()

# Localiza dentro da base de dados quais os dados estão com valor null
base_credit.loc[pd.isnull(base_credit['age'])]
base_credit.loc[base_credit['age'].isnull()]

# Novamente vamos adotar a estratégia de preencher estes valores com a média
# fillna é uma função do pandas que preenche valores NaN (nulos)
# inplace indica para alterar o valor na base, caso esteja como falso irá alterar somente na memória
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)

# Vamos olhar os novos dados
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])
