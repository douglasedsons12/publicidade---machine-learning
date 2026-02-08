
# %%
# Importando bibliotecas

import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# 
# %%
# Importando dados

df = pd.read_csv('data_messy.csv')

# %%
# Printando nome das colunas
print(df.columns.tolist())
# %%
# Limpando títulos das colunas
df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
# %%
# Limpando coluna: de string para float

colunas_sujas = df['spend'].astype(str).str.contains(r'\$')
print(df.loc[colunas_sujas,['campaign_id','spend']].head(3))
df['spend'] = df['spend'].astype(str).str.replace(r'[^\d.-]','',regex=True).astype(float)
# %%
# Limpando listas

df['channel'].unique().tolist()

dicionario = {
    'Tik_Tok':'TikTok',
     'Facebok':'Facebook',
    'E-mail':'Email',
    'Insta_gram':'Instagram',
     'Gogle': 'Google Ads',
        'N/A':np.nan
}

df['channel'] = df['channel'].replace(dicionario)
df['channel'].unique().tolist()

# %%
# Manuseando booleans

print(df['active'].unique())


boolean = {
    'Y': True,
    0: False,
    1: True,
    'No': False,
    'Yes': True,
    '1':True,
    '0': False,
    'True': True,
    'False': False

}

df['active'] = df['active'].map(boolean).fillna(False).astype(bool)
print(df['active'].unique())
# %%
# Convertendo para data os valores em string

df['start_date'] = pd.to_datetime(df['start_date'],errors='coerce',dayfirst=False)
df['end_date'] = pd.to_datetime(df['end_date'],errors='coerce',dayfirst=False)

print(df['end_date'].dtype)

# %%
# Removendo colunas duplicadas
df = df.loc[:, ~df.columns.duplicated()]

# %%
# Integridade Lógica
impossible_mask = df['clicks'] > df['impressions']
print(df.loc[impossible_mask,['campaign_id','impressions','clicks']].head(3))
# %%
# Evitando viagem no tempo
time_travel = df['end_date'] < df['start_date']
df.loc[time_travel,'end_date'] = df.loc[time_travel,'start_date'] + pd.Timedelta(days=30)
print(df.loc[time_travel,['campaign_id','start_date','end_date']].head(3))


# %%
# Lidando com gastos negativos
negativo = df['spend'] < 0
print(df.loc[negativo,['spend','campaign_id']].head(3))
mediana= df.loc[df['spend'] >= 0,'spend'].median()
df.loc[negativo,'spend'] = mediana

# %%
# Manuseando outliers

Q1 = df['spend'].quantile(0.25)
Q3 = df['spend'].quantile(0.75)
IQR = Q3 - Q1
upper_limite = Q3 + (1.5*IQR)

outlier_mask = df['spend'] > upper_limite
print(df.loc[outlier_mask,['spend','campaign_id']].head(3))

df.loc[outlier_mask,'spend']= upper_limite
print(df.loc[outlier_mask,['spend','campaign_id']].head(3))

# %%
sb.boxplot(x=df['conversions'])
# %%
# Extraindo strings

df['season'] = df['campaign_name'].str.extract(r'Q\d_([^_]+)_')


# %%
# Salvando em PDF
df.to_csv('publicidade.csv',index=False,header=True,sep= ';')
# %%
# Limpeza em outros atributos

df = df.dropna(subset=['conversions']).copy()
df['channel'] = df['channel'].fillna('unknown')
df['season'] = df['season'].fillna('unknown')
# %%
# Criação de novas colunas

df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
df['duration_days'] = df['duration_days'].fillna(df['duration_days'].median())

# %%
# Dummização

features = ['spend','channel','duration_days','clicks','spend','season']
X = df[features]
Y = df['conversions']

X = pd.get_dummies(X,columns=['channel','season'])


# %%
# Treinamento do projeto

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2,random_state = 42)

modelo = RandomForestRegressor(n_estimators=100, random_state=42)

modelo.fit(X_train,Y_train)
# %%
# Avaliação do modelo

previsoes = modelo.predict(X_test)
print(f"R² Score: {r2_score(Y_test, previsoes):.4f}")
print(f"Erro Médio: {mean_absolute_error(Y_test, previsoes):.2f}")
# %%
importance = pd.Series(modelo.feature_importances_,index = X.columns)
importance.sort_values().plot(kind='barh',color='steelblue')
plt.title('Importância das Variáveis')
# %%
# --- GRÁFICO DE DISPERSÃO (REAL vs PREVISTO) ---
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, previsoes, alpha=0.5, color='orangered')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)

plt.title('Real vs. Previsto')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')

# SALVAR NO DISCO
plt.savefig('grafico_dispersao.png', dpi=300, bbox_inches='tight')
plt.show()