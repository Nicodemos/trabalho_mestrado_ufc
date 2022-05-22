from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle as pk
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)
data=pd.read_csv('Fortaleza_dataset_v3.csv',
                                  sep=',',
                                  encoding = 'utf8', parse_dates=['data_notifica'])

atributos_selecionados = ['data_notifica', 'Dengue','Dengue_sma7','Acumulado_21','Acumulado','Populacao','Densidade_Dem','Precipitacao_sma7','Umidade_7sma','Vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']
x = data[atributos_selecionados].values

# configurações da rede
look_back = 1
otimizador='Adam'
otimizador2 = 'RMSprop'
metricas=['mean_absolute_error']

for i in range(len(alvo)):

    print('semana: ',data[alvo[i]])
    y = data[alvo[i]].values
    #y = data['T5'].values
    metr = ['mean_absolute_error']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=20)

    xtrain_sem_date = xtrain[:,1:]
    xtest_sem_date = xtest[:,1:]

    # dados de treino padronizados
    standardx = StandardScaler()
    standardy = StandardScaler()
    training_set_scaledx = standardx.fit_transform(xtrain_sem_date)
    training_set_scaledy = standardy.fit_transform(np.array(ytrain).reshape(-1,1))

    # Treino, 9 camadas
    model = Sequential()
    model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(Dense(units=70,activation = 'relu'))
    model.add(Dense(units=100,activation = 'relu'))
    model.add(Dense(units=80, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_absolute_error', optimizer=otimizador, metrics=metricas)

    inicio = timeit.default_timer()  # Inicio - medição de tempo de execução
    model.fit(training_set_scaledx, training_set_scaledy, epochs=250, batch_size=10)
    fim = timeit.default_timer()
    print('Duração: ',fim - inicio)
    model.summary()

    y_pred = standardy.inverse_transform((model.predict(standardx.transform(xtest_sem_date))))
    y_pred = np.array(y_pred).reshape(-1,1)
    ytest = np.array(ytest).reshape(-1,1)
    print('PredicaoR2: ',r2_score(y_pred,ytest)*100)
    print('PredicaoMAE: ',mean_absolute_error(y_pred,ytest))

    date = xtest[:,:1] # pegar data para utilizar como index
    y_pred = np.array(y_pred).reshape(-1,1)
    ytest = np.array(ytest).reshape(-1,1)
    data_f = pd.DataFrame(y_pred, columns=['Predito'])
    data_f['Real'] = ytest
    data_f['Data'] = date
    data_f.set_index('Data',inplace=True)
    print(data_f['Predito'].plot(linestyle='dashed',color="#FF5733",fontsize=12))
    print(data_f['Real'].plot(linestyle='solid',color="#080A48",fontsize=12))
    plt.tight_layout()
    plt.legend(loc='upper right',fontsize=12)
    plt.show()

    ## guardando o modelo treinado
    model.save('..\\modelos_para_uso\\',overwrite=True)
