# DIVISAO ENTRE PREVISORES E CLASSE -  ESCALONAMENTO DOS DADOS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos
import pickle
# Pickle nos auxilia para salvar a base de dados que utilizaremos

base_credit = pd.read_csv(
    '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/credit_data.csv')
# Trata dos dados errados
mediaIdades = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = mediaIdades
# Trata dos dados nulos
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)


X_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values


# Padronização dos dados é indicado para quando há valores muitos distantes uns dos outros
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(
    X_credit, y_credit, test_size=0.25, random_state=0)

print(X_credit_treinamento.shape)

# Salvar em disco essas bases já separadas
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento,
                X_credit_teste, y_credit_teste], f)
