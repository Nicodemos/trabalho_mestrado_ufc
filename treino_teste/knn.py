from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import timeit
import pickle as pk
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)
data=pd.read_csv('Fortaleza_dataset_v3.csv',
                                  sep=',',
                                  encoding = 'utf8', parse_dates=['data_notifica'])

atributos_selecionados = ['data_notifica', 'Dengue','Dengue_sma7','Acumulado_21','Acumulado','Populacao','Densidade_Dem','Precipitacao_sma7','Umidade_7sma','Vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

x = data[atributos_selecionados].values

for i in range(len(alvo)):
    print('semana: ',data[alvo[i]])
    y = data[alvo[i]].values
    #y = data['T5'].values
    metr = ['mean_absolute_error']
    neigh = KNeighborsRegressor(n_neighbors=6, metric='euclidean') # definição do hiperpârametros
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=20)
    print("shape: ",xtrain.shape)
    xtrain_sem_date = xtrain[:,1:]
    xtest_sem_date = xtest[:,1:]

    # Padroniza os dados de treino, padronização é diferente de normalização, na prática tentam resolver o mesmo problema
    scalerx = StandardScaler()
    scalery = StandardScaler()


    standardx = scalerx.fit_transform(xtrain_sem_date)
    standardy = scalery.fit_transform(np.array(ytrain).reshape(-1,1))

    inicio = timeit.default_timer()  # Inicio - medição de tempo de treino
    neigh.fit(standardx, standardy)
    fim = timeit.default_timer()

    print('Duração: ',fim - inicio)
    teste = neigh.score(standardx, standardy)
    print('Score Teste',teste)

    y_pred = scalery.inverse_transform ((neigh.predict(scalerx.transform(xtest_sem_date))))
    print('R²: ',r2_score(y_pred, ytest)*100)
    print('MAE: ',mean_absolute_error(y_pred, ytest))

    array_to_graf = [xtest[:,:1]]
    ypred = pd.DataFrame(y_pred, columns=['Predito'])
    ytest = pd.DataFrame(ytest, columns=['Real'])
    dta = pd.DataFrame(np.squeeze(array_to_graf), columns=['Data'])

    dta['Predito'] = ypred
    dta['Real'] = ytest
    dta.set_index('Data',inplace=True)
    all_dta_to_graf = dta.sort_index()
    print(all_dta_to_graf)
    all_dta_to_graf['Predito'].plot(linestyle='dashed',color="#FF5733",fontsize=12)
    all_dta_to_graf['Real'].plot(linestyle='solid',color="#080A48",fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right',fontsize=12)
    plt.show()

## guardando o modelo treinado
    #with open('..\\modelos_para_uso\\knn.pkl', 'wb') as file:
        #pk.dump(neigh, file)
