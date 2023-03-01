# Divisão entre previsores e classe

# import das bibliotecas
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

# Tratamento dos dados categóricos

# LabelEncoder
# Quando necessitamos de transformar labels no formato String, por exemplo, para números a fim de por utilizá-los nos algoritmos de machine learning
label_encoder_teste = LabelEncoder()
# Vamos testar na coluna workclass
teste = label_encoder_teste.fit_transform(X_census[:, 1])
# print(teste)
# Note que os valores únicos dessa coluna foram substituidas por números

# Dessa forma
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

# print(X_census[0])
# Note que agora possuimos apenas valores númericos para a base de dados

# OneHotEncoder
# Evita que o mapeamento dos dados atrapalhe os algoritmos, uma vez que este pode entender que um atributo é mais importante que outro, visto sua identificação alta


oneHotEncoder_census = ColumnTransformer(transformers=[(
    'OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
# Fornecemos o métofo, as colunas que queremos mudar e o remainder que indica que queremos manter as outras colunas
X_census = oneHotEncoder_census.fit_transform(X_census).toarray()
print(X_census[0])
