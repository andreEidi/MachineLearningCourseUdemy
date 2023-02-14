# Escalonamento dos dados

# import das bibliotecas
import pickle
# picke é uma biblioteca que nos permite salvar o banco
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # graficos dinamicos

# exploração inicial dos dados
base_census = pd.read_csv(
    '/home/andre/Projetos/MachineLearningCourseUdemy/Bases_de_dados/census.csv')

X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

label_encoder_workclass = LabelEncoder()
X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])

label_encoder_education = LabelEncoder()
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])

label_encoder_marital = LabelEncoder()
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])

label_encoder_occupation = LabelEncoder()
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])

label_encoder_relationship = LabelEncoder()
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])

label_encoder_race = LabelEncoder()
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])

label_encoder_sex = LabelEncoder()
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])

label_encoder_country = LabelEncoder()
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

# OneHotEncoder
# Evita que o mapeamento dos dados atrapalhe os algoritmos, uma vez que este pode entender que um atributo é mais importante que outro, visto sua identificação alta

oneHotEncoder_census = ColumnTransformer(transformers=[(
    'OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
# Fornecemos o métofo, as colunas que queremos mudar e o remainder que indica que queremos manter as outras colunas
X_census = oneHotEncoder_census.fit_transform(X_census).toarray()

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    X_census, y_census, test_size=0.15, random_state=0)

print(X_census_treinamento.shape)

# Salvar em disco essas bases já separadas
with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento,
                X_census_teste, y_census_teste], f)
