# DIVISAO ENTRE PREVISORES E CLASSE -  ESCALONAMENTO DOS DADOS

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

base_credit = pd.read_csv(
    '/home/andre/Projetos/Machine_learning_udemy_course/Bases_de_dados/credit_data.csv')
# Trata dos dados errados
mediaIdades = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = mediaIdades
# Trata dos dados nulos
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)


X_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

# note que os valores estão muito discrepantes, em escalas diferentes
lista_minimos = [X_credit[:, 0].min(), X_credit[:, 1].min(),
                 X_credit[:, 2].min()]
print(lista_minimos, end='\n\n')

# Padronização dos dados é indicado para quando há valores muitos distantes uns dos outros
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
Nova_lista_minimos = [X_credit[:, 0].min(), X_credit[:, 1].min(),
                      X_credit[:, 2].min()]
print(Nova_lista_minimos, end='\n\n')
