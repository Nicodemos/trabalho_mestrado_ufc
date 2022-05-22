from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
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

metricas=['mean_absolute_error']

for i in range(len(alvo)):

    print('semana: ',data[alvo[i]])
    y = data[alvo[i]].values
    #y = data['T5'].values
    metr = ['mean_absolute_error']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=20)

    xtrain_sem_date = xtrain[:,1:]
    xtest_sem_date = xtest[:,1:]

    standardx = StandardScaler()
    standardy = StandardScaler()

    # padronizando os dados de treino
    training_set_scaledx = standardx.fit_transform(xtrain_sem_date)
    training_set_scaledy = standardy.fit_transform(np.array(ytrain).reshape(-1,1))

    # modelando os dados de treino
    training_set_scaledx = training_set_scaledx.reshape(-1,1,len(atributos_selecionados[1:]))
    training_set_scaledy = training_set_scaledy.reshape(-1,1,1)
    training_set_scaledx_t = xtest_sem_date.reshape(-1,1, len(atributos_selecionados[1:]))

    # Treino, 9 camadas
    # kernel_initializer => Inicializa os pesos com o método he_uniform
    # activation => função de ativação para cada neurônio de uma camada
    # input_shape => o forma como os dados vão entrar no rede, uma linha com 9 tipos de atibutos por vez (uma instância)
    model = Sequential()
    model.add(LSTM(100,activation = 'relu',input_shape=(1,9), kernel_initializer = 'he_uniform',return_sequences=True))
    model.add(Dense(units=70, activation = 'relu'))
    model.add(Dense(units=100, activation = 'relu'))
    model.add(Dense(units=80, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=90, activation = 'relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=metricas)

    inicio = timeit.default_timer()  # Inicio - medição de tempo de execução
    model.fit(training_set_scaledx, training_set_scaledy, epochs=250, batch_size=10)
    fim = timeit.default_timer()
    print('Duração: ',fim - inicio)
    model.summary()

    # padronizado os dados de entrada
    standard_ = StandardScaler()
    standard_test = standard_.fit_transform(xtest_sem_date)
    standard_test = standard_test.reshape(-1,1,9) # modela os dados de entrada, 9 tipos  de atributos

    y_pred = standardy.inverse_transform(np.array(model.predict(standard_test)).reshape(-1,1))
    print('PredicaoR2: ',r2_score(np.squeeze(y_pred),ytest)*100)
    print('PredicaoMAE: ',mean_absolute_error(np.squeeze(y_pred),ytest))

    array_to_graf = [xtest[:,:1]]
    ypred = pd.DataFrame(np.squeeze(y_pred), columns=['Predito'])
    ytest = pd.DataFrame(np.squeeze(ytest), columns=['Real'])
    dta = pd.DataFrame(np.squeeze(array_to_graf), columns=['Data'])

    dta['Predito'] = ypred
    dta['Real'] = ytest
    dta.set_index('Data',inplace=True)
    all_dta_to_graf = dta.sort_index()
    all_dta_to_graf['Predito'].plot(linestyle='dashed',color="#FF5733",fontsize=12)
    all_dta_to_graf['Real'].plot(linestyle='solid',color="#080A48",fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right',fontsize=12)
    plt.show()
