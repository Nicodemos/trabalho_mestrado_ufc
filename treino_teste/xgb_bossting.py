import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle as pk
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

data=pd.read_csv('Fortaleza_dataset_v3.csv',
                                  sep=',',
                                  encoding = 'utf8',parse_dates=['data_notifica'])

atributos_selecionados = ['data_notifica', 'Dengue','Dengue_sma7','Acumulado_21','Acumulado','Populacao','Densidade_Dem','Precipitacao_sma7','Umidade_7sma','Vento_mps_7sma']
alvos=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

x = data[atributos_selecionados]
y = data['T5']
print("xgb: ",y)

# Modelo XGBoosting
xgbr_model = xgb.XGBRegressor(max_depth=7,seed=10,learning_rate=0.2,booster='gbtree')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=20)

# padronizando os dados de treino
normalizax = StandardScaler()
normalizay = StandardScaler()

xtrain = np.array(xtrain.iloc[:, 1:]).reshape(-1,len(atributos_selecionados[1:])) # remodelando os dados: linhas e colunas
ytrain = np.array(ytrain).reshape(-1,1)

# Treino
model = xgbr_model.fit(xtrain, ytrain)

xtest_com_date = xtest # na próxima linha seleciono apenas os valores sem a data.
xtest = np.array(xtest.iloc[:, 1:]).reshape(-1,len(atributos_selecionados[1:])) # remodelando os dados: linhas e colunas

ypredict = model.predict(xtest)
print('Predição MAE: ',mean_absolute_error(ypredict,ytest))
print('Predição R2: ',r2_score(ypredict,ytest)*100)

data_f = pd.DataFrame(ypredict,columns=['Predito'])
ytest = np.array(ytest).reshape(-1,1)

data_f['Data'] = np.array(xtest_com_date.iloc[:,0:1]).reshape(-1,1)
data_f['Real'] = ytest

data_f.set_index('Data',inplace=True)
all_dta_to_graf = data_f.sort_index()
all_dta_to_graf['Predito'].plot(linestyle='dashed',color="#FF5733",fontsize=12)
all_dta_to_graf['Real'].plot(linestyle='solid',color="#080A48",fontsize=12)
plt.tight_layout()
plt.legend(loc='upper right',fontsize=12)
plt.show()

# Salvando o modelo xgbbr
with open('..\\modelos_para_uso\\xgboost.pkl', 'wb') as file:
    pk.dump(model, file)
