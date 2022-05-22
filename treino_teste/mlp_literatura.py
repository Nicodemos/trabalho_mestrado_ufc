#!/usr/bin/env python
# coding: utf-8

# <h2>Modelo de Rede Neural Encadeada Multi-link para predição de infectados por dengue na rede de atenção básica de saúde.</h2>

# In[1]:


import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import seaborn as sns

import os

import pandas as pd
pd.set_option('display.max_rows', 100)

from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,matthews_corrcoef

import time
from datetime import datetime, timedelta

from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

# Tipo e versão do modelo
tipo='MLP_Multi_link_chaining'

# Versão do dataset: apenas auditados (contendo dados para I1, I2, I3,..., I10).
versao='v1'

cidade='Fortaleza'

cidades={1:'Fortaleza'}


# <h4> Carregamento e Preparação dos dados </h4>

# In[2]:


''' Preparação dos dados.
'''

data=None

for cid in cidades:
    data=(pd.read_csv('Fortaleza_dataset_v3.csv',
                                  sep=',',
                                  encoding = 'utf8', parse_dates=['data_notifica']))

# Renomeia as colunas
data.columns=['date','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15','Daily Infecteds','Infecteds sma7', 'Cumulated 21 days','Cumulated', 'Population','Rt week', 'Susceptible', 'Population Den.', 'Precipitation sma7', 'Avg. Temperature sma7','Min. Temperature sma7', 'Humidity sma7','Wind Speed sma7','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

#Lista de todas as variáveis independentes.
lista_variaveis=['Daily Infecteds','Infecteds sma7', 'Cumulated 21 days','Cumulated', 'Population','Rt week', 'Susceptible', 'Population Den.', 'Precipitation sma7', 'Avg. Temperature sma7','Min. Temperature sma7', 'Humidity sma7','Wind Speed sma7']

# Lista de atributos selecionados e alvos
selecao_var=['Infecteds sma7', 'Cumulated 21 days', 'Rt week', 'Susceptible', 'Population Den.', 'Population', 'Precipitation sma7', 'Avg. Temperature sma7', 'Humidity sma7','Wind Speed sma7']
selecao_alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']
selecao_confirmados=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11','C12', 'C13', 'C14', 'C15']

#data.reset_index(drop=True, inplace=True)
#data.set_index('date', inplace=True)

data.head()


# In[3]:


# Analisa os dados em busca de Not a Number (NaN) ou valores nulos.
total_nulos=data.isnull().sum()
total_nulos


# Gráfico de Correlação das entradas escolhidas.

print(selecao_var)



#Quantidade de alvos.
n_alvos=len(selecao_alvo)

# adicionado by Nicodemos
selecao_mod = []
selecao_mod = selecao_alvo
selecao_mod.append('date')
x = pd.DataFrame(data, columns=selecao_var)
y = pd.DataFrame(data, columns=selecao_mod)

print('Valor x: ',x.head(10))
print('valor y: ',y.head())


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=20)

aux = pd.DataFrame(y_test)
aux.set_index('date',inplace=True)
date_index = aux.index

y_test = pd.DataFrame(y_test)
y_test['index'] = [x for x in range(0,len(y_test))]
y_test.set_index('index',inplace=True)


y_train = y_train.iloc[:,:-1]

print("y_test =>>>>>>>>> \n", y_test)
print("y_train =>>>>>>>>> \n", y_train)
#print('date_index => \n',date_index)


# Número de épocas
num_epochs=250

#Escolhendo o número de neurônios na camada escondida.

#Ni = number of input neurons (número de entradas no dataset).
Ni= X_train.shape[1]

#No = number of output neurons.
No=1

#Ns = number of samples in training data set (?).
Ns=X_train.shape[0]

#a = an arbitrary scaling factor usually 2-10 (começaremos por 6).
a=10

Nh = int(np.round(Ns/(a*(Ni + No)),0))

print('Número de neurônios ocultos: ',Nh)

# Quantidade de amostras por lote
num_batch=10

# Porcentagem de amostras para validação do modelo enquanto treina.
validation=0.2

# Define a métrica de avaliação.
# 'root_mean_squared_error'
# 'mean_squared_error'
# 'mean_absolute_error'
# 'mean_absolute_percentage_error'
# 'mean_squared_logarithmic_error'
# 'cosine_similarity'
metrica=['mean_absolute_error']

# Define a função erro considerada no back-porpagation.
#'root_mean_squared_error'
#'mean_squared_error'
#'mean_absolute_error'
funcao_loss= 'mean_absolute_error'

# Define o otimizador para atualização dos pesos.
# 'RMSprop'
# 'Adam'
# 'SGD'
# 'Adagrad'
# 'Adadelta'
# 'Adamax'
# 'Nadam'
# 'Ftrl'
otimizador='Adam'


# <h4>Defeinição da arquitetura e treino dos modelos</h4>

# In[24]:


# Lista contendo o Loss médio de cada modelo semanal.
loss_medio_target_mlnn=[]

#lista de Modelos em cascata.
modelo_cascata=[]

#Lista de dados de treino, validação e teste cascata.
X_train_cascata=[]
X_test_cascata=[]
#X_val_cascata=[]

selecao_treino=selecao_var.copy()

#X_train_montado=X_train.copy()
#X_test_montado=X_test.copy()

X_train_montado=X_train.copy()
X_test_montado=X_test.copy()

for target in range(n_alvos):
    # Compõe os dados para treino, validação e teste

    ''' Normaliza a entrada
    '''

    # Normalizador estatístico, onde o resultado da normalização garante média zero e variância/desvio parão unitária.
    normalizador=StandardScaler()

    #print('X_train_montado: \n',X_train_montado)
    #print('Tamanho X_train_montado: \n',len(X_train_montado))

    X_train_cascata.append(pd.DataFrame(normalizador.fit_transform(np.array(X_train_montado)),columns=selecao_treino))
    X_test_cascata.append(pd.DataFrame(normalizador.transform(np.array(X_test_montado)),columns=selecao_treino))
    #X_val_cascata.append(pd.DataFrame(normalizador.transform(np.array(X_val_montado)),columns=selecao_treino))

    # Grava normalizador
    if not os.path.exists('MODEL\\'+tipo+'_'+versao):
        os.makedirs('MODEL\\'+tipo+'_'+versao)

    # Grava normalizador
    dump(normalizador, 'MODEL\\'+tipo+'_'+versao+'\\'+'normalizador_S'+str(target+1)+'.bin', compress=True)

    # Inicializando o modelo: L.
    model_Sx = Sequential(name='Model_T'+str(target+1))

    # Adicionando a camada de entrada
    model_Sx.add(Dense(units = Nh, activation = 'relu', input_dim = X_train_cascata[target].shape[1], kernel_initializer = 'he_uniform'))

    #Dropout de 80%
    #model_Sx.add(Dropout(0.8))

    # Adicionando a primeira camada ocultarange
    model_Sx.add(Dense(units = Nh, activation = 'relu'))

    # Adicionando a camada de saída
    model_Sx.add(Dense(units = 1))


    # Compilando o modelo S.
    model_Sx.compile(optimizer = otimizador, loss = funcao_loss, metrics=metrica)
    #print('xtrain cscata=: ',np.array(X_train_cascata[target]))
    #print('xtrain cscata=: ',np.array(y_train.iloc[:,target]))

    #xtrein = np.array(X_train_cascata[target])
    #ytrein = np.array(y_train.iloc[:,target])




    print("xtrein : \n",X_train_cascata[target])
    print("ytrein : \n",y_train.iloc[:,target])
    print("-------------------------------------------------")
    print("tamanho y : ",len(y_train.iloc[:,target]))
    print("tamanho x : ",len(X_train_cascata[target]))

    history_Sx=model_Sx.fit(np.array(X_train_cascata[target]),
                            np.array(y_train.iloc[:,target]),
                            batch_size = num_batch,
                            epochs = num_epochs,
                            #validation_split=validation,
                            shuffle=True,
                            verbose=0,)


    ''' Gravação do modelo.
    '''

    modelo_cascata.append(model_Sx)

    # Grava os modelos treinados
    if not os.path.exists('MODEL\\'+tipo+'_'+versao):
        os.makedirs('MODEL\\'+tipo+'_'+versao)

    model_Sx.save('MODEL\\'+tipo+'_'+versao+'\\modelS'+str(target+1))


    print('\n# Avaliando loss com dados de teste:')
    loss_Sx_teste = model_Sx.evaluate(np.array(X_test_cascata[target]), np.array(y_test.iloc[:,target]), batch_size=10)
    model_Sx.summary()

    # Faz estatística de loos médio por target.
    loss_medio_target_mlnn.append(np.round(loss_Sx_teste[0],decimals=2))

    print('test loss modelo '+str(target+1)+': {}'.format(loss_medio_target_mlnn[target]))

    # "Plot Training Validation & Test Loss"
    # plt.figure(figsize=(10,5))
    #
    # plt.title('Model Loss - T'+str(target+1))
    # plt.plot(history_Sx.history['loss'][:num_epochs],linestyle='-',color = '0', label='Train')
    # plt.plot(history_Sx.history['val_loss'][:num_epochs],linestyle='-.',color = '0', label='Validation')
    # plt.axhline(y=loss_Sx_teste[0],linestyle=':',color = '0', label='Test')
    # #plt.ylabel(metrica[0])
    # plt.ylabel('MAE')
    # plt.xlabel('ephoc')
    # #plt.ylim(0,60)
    # plt.legend(loc='upper right')
    # plt.show()

    y_pred_Sx = model_Sx.predict(np.array(X_test_cascata[target]))


    #print('y_predito: ==>',y_pred_Sx)

    # Plota resultado para os dados de teste

    # Ordena os índices de resultados
    index = np.array(y_test.iloc[:,target]).argsort()

    # plt.figure(figsize=(20,6))
    # plt.plot(np.array(y_test.iloc[index,target]), color = 'red', label = 'Real')
    # plt.plot(y_pred_Sx[index], marker='.', lw=0, fillstyle='none', color = 'blue', label = 'Predicto')
    # plt.title('Predição para semana '+str(target+1))
    # plt.grid()
    # plt.legend()
    # plt.show()


    ''' Aplicação do modelo nos dados do dataset para criação da predição P_alvo
    '''

    # Predição infecção por dengue nas próximas semanas.
    # Não considera predições negativas, caso existam.
    y_pred_train_Sx=np.maximum(0,np.round(model_Sx.predict(np.array(X_train_cascata[target])),decimals=0))
    y_pred_test_Sx=np.maximum(0,np.round(model_Sx.predict(np.array(X_test_cascata[target])),decimals=0))
    #y_pred_val_Sx=np.maximum(0,np.round(model_Sx.predict(np.array(X_val_cascata[target])),decimals=0))

    data_graf = pd.DataFrame(y_pred_test_Sx, columns=['Predito'])

    print('y_test.iloc[:,target]: ',y_test.iloc[:,target])
    data_graf['Real'] = y_test.iloc[:,target]
    data_graf['Data'] = date_index
    data_graf.set_index('Data',inplace=True)
    data_graf = data_graf.sort_index()
    data_graf['Predito'].plot(linestyle='dashed',color="#FF5733",fontsize=12)
    data_graf['Real'].plot(linestyle='solid',color="#080A48",fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right',fontsize=12)
    plt.show()






    # Concatena o resultado do respectivo modelo na matriz infectados, coluna Ipx, x={1,2,3,...,10}.
    #X_train_montado = pd.DataFrame(X_train_montado)
    X_train_montado['Ip'+str(target+1)] = y_pred_train_Sx[:,:]
    print('y_pred_train_Sx \n',y_pred_train_Sx)
    print('Tamanho y_pred_train_Sx \n',len(y_pred_train_Sx))

    X_test_montado['Ip'+str(target+1)] = y_pred_test_Sx
    #X_train_montado=pd.concat([X_train_montado,pd.DataFrame(y_pred_train_Sx, columns=['Ip'+str(target+1)])],axis=1)
    #X_test_montado=pd.concat([X_test_montado,pd.DataFrame(y_pred_test_Sx, columns=['Ip'+str(target+1)])],axis=1)
    #X_val_montado=pd.concat([X_val_montado,pd.DataFrame(y_pred_val_Sx[:], columns=['Ip'+str(target+1)])],axis=1)

    selecao_treino.append('Ip'+str(target+1))

# Plota gráfico de Loss por modelo Semanal.

# plt.figure(figsize=(20,6))
# plt.plot(np.arange(1, n_alvos+1, 1.0),loss_medio_target_mlnn, color = 'red', label = 'Loss')
# plt.title('Loss Teste x target')
# plt.grid()
# plt.xticks(np.arange(1, n_alvos+1, 1.0))
# plt.legend()
# plt.show()


# <h4> Avaliando para todo o conjunto de dados. </h4>

# In[25]:


# Lista com valores R2 para cada semana predita
r2_mlnn=[]

# Separando variáveis independentes e alvos.
X_=data.reset_index().loc[:,selecao_var].copy()
y_=data.reset_index().loc[:,selecao_alvo].copy()

# Cria a lista para selecionar as colunas
selecao_producao=selecao_var.copy()

for target in range(len(selecao_alvo)):

    # Reestabelece o normalizador configurado a partir do arquivo.
    normalizador=load('..\\MODEL\\'+tipo+'_'+versao+'\\'+'normalizador_S'+str(target+1)+'.bin')

    # Restabelece o modelo treinado a partir do arquivo.
    model_Sx =tf.keras.models.load_model('..\\MODEL\\'+tipo+'_'+versao+'\\S'+str(target+1))

    # Normaliza a entrada da RNA
    X=pd.DataFrame(normalizador.transform(np.array(X_)),columns=selecao_producao)

    # print(len(X))

    # Predição infecção por dengue nas próxima Sx semana.
    # Não considera predições negativas, caso existam.
    y_pred_Sx=np.maximum(0,np.round(model_Sx.predict(np.array(X)),decimals=0))

    print(len(y_pred_Sx))

    #Cálculo das correlações:
    r2_mlnn.append(r2_score(np.array(y_.iloc[:,target]), y_pred_Sx))

    print('Número de amostras de '+cidades[cid]+': {}'.format(X.shape[0]))
    print('Número de característicasde: {}'.format(X.shape[1]))
    print('R2: {}'.format(r2_score(np.array(y_.iloc[:,target]), y_pred_Sx)))
    print('Correlação Pearson: {}'.format(matthews_corrcoef(np.array(y_.iloc[:,target]), y_pred_Sx)))

    # plt.figure(figsize=(15,6))
    # plt.plot(np.array(y_.iloc[:,target]),'-', color = 'red', label = 'Real')
    # plt.plot(y_pred_Sx,'.', color = 'blue', label = 'Predito')
    # plt.title('Predição para '+cidades[cid]+' '+str(target+1)+' semanas a frente')
    # plt.ylabel('Casos')
    # plt.xlabel('Semana')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Concatena o resultado do respectivo modelo na matriz infectados, coluna Ipx, x={1,2,3,...,10}.
    X_=pd.concat([X_,pd.DataFrame(y_pred_Sx[:], columns=['P'+str(target+1)])],axis=1)

    print(len(X_))

    #Inclui nova coluna para próxima MLP em cascata
    selecao_producao.append('P'+str(target+1))

# Concatena datas e infecções na data.
dataset=pd.concat([data[['Daily Infecteds']+selecao_confirmados].reset_index(),X_],axis=1)

#Concatena as variáveis alvo, montando o dataset com as predições realizadas.
dataset=pd.concat([dataset,y_],axis=1)
dataset.set_index('date',inplace=True)
dataset.index=pd.to_datetime(dataset.index)

# Grava os dados de predição
if not os.path.exists('..\\PREDICTION\\'+tipo+'_'+versao):
    os.makedirs('..\\PREDICTION\\'+tipo+'_'+versao)

# Salva os dados preditos no respectivo arquivo.
dataset.to_csv('..\\PREDICTION\\'+tipo+'_'+versao+'\\infectados_predito_'+cidades[cid]+'.csv',index=False)


# <h4> Seleção das semanas epidemiológicas </h4>

# In[26]:


dataset


# In[27]:


# Realiza uma cópia para trabalhar com dados semanais, considerando o fim de toda semana epidemiológica nos sábados.
dataset_semanal=dataset.copy()

# Seleciona apenas os registros de início de semana epidemiológica.
# Considerando as semanas epidemiológicas começando de domingo (day of week == 0) indo até sábado (day of week == 6).
# Acumular os valores ao sábado, pois considera o somatório da semana toda.
dataset_semanal=dataset_semanal.loc[dataset_semanal.index.dayofweek==6,:]

dataset_semanal


# <h4> Funções Sigmóide, primeira derivada, segunda derivada e regressão logística </h4>

# In[28]:


def sigmoide(x, L ,x0, k):
    ''' Retorna o resultado da função logística (curva sigmoide), onde
        L: número de casos total da epidemia em análise;
        x: semana em questão;
        x0: semana onde ocorre o pico da epidemia;
        k: ???
    '''
    return L / (1 + np.exp(-k*(x-x0)))

def sigmoide_derivada1(x, L ,x0, k):
    ''' Retorna o resultado da função logística (curva sigmoide), onde
        L: número de casos total da epidemia em análise;
        x: semana em questão;
        x0: semana onde ocorre o pico da epidemia;
        k: ???
    '''
    return L*k* sigmoide(x, 1,x0,k)*(1-sigmoide(x,1,x0,k))

def sigmoide_derivada2(x, L ,x0, k):
    ''' Retorna o resultado da função logística (curva sigmoide), onde
        L: número de casos total da epidemia em análise;
        x: semana em questão;
        x0: semana onde ocorre o pico da epidemia;
        k: ???
    '''
    return L*(k**2)*sigmoide(x, 1,x0,k)*(1-sigmoide(x,1,x0,k))*(1-2*sigmoide(x,1,x0,k))

def regressao_sigmoide(serie_infectados):
    ''' Entrada: lista de infectados semanal/diário/etc.
        Retorna: lista de infectados semanal/diário/etc resultado de regressão sigmóide.
    '''

    n=len(serie_infectados)

    if (n == 0):
        return 0
    else:
        # Valores do eixo y.
        ydata = serie_infectados

        # Valores do eixo x.
        xdata = np.linspace(1,n,n)

        # Faz regressão de acordo com a função sigmoide_derivada1
        (L , x0, k), pcov = curve_fit(sigmoide,
                                      xdata,
                                      ydata,
                                      p0=[max(ydata), np.median(xdata),1],
                                      maxfev = 2500)

        # retorna a série de dados.
        return sigmoide(xdata, L, x0, k)

def regressao_sigmoide_derivada1(serie_infectados):
    ''' Entrada: lista de infectados semanal/diário/etc.
        Retorna: lista de infectados semanal/diário/etc resultado de regressão para a curva derivada da sigmóide.
    '''

    n=len(serie_infectados)

    if (n == 0):
        return 0
    else:
        # Valores do eixo y.
        ydata = serie_infectados

        # Valores do eixo x.
        xdata = np.linspace(1,n,n)

        try:
            # Faz regressão de acordo com a função sigmoide_derivada1
            (L , x0, k), pcov = curve_fit(sigmoide_derivada1,
                                          xdata,
                                          ydata,
                                          p0=[max(ydata), np.median(xdata),1],
                                          maxfev = 2500)
            # retorna a série de dados.
            return sigmoide_derivada1(xdata, L, x0, k)
        except:
            # retorna a série de dados.
            return ydata


# <h4> Funções de Plotagem e Acompanhamento do Modelo </h4>

# In[29]:


# Função para plotar a predição para um dia específico, sem regressão.

dia='2020-03-08'

def plota_grafico_dia(dia, dataset_predicao):
    ''' Aceita dia como datetime ou dia no formato "%Y-%m-%d".
    '''
    if not isinstance(dia, datetime):
        dia=datetime.strptime(dia, "%Y-%m-%d")

    antes=dataset_predicao.loc[dia,['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15']]
    depois=dataset_predicao.loc[dia,['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15']]

    janela=pd.concat([antes,depois])

    # plt.figure(figsize=(10,6))
    #
    # plt.bar(range(1, len(antes)+1),antes, label='Casos confirmados')
    # plt.bar(range(len(antes)+1,len(antes)+len(depois)+1),depois,label='Predito 15', color='navajowhite')
    #
    # plt.title('Fortaleza: '+dia.strftime("%d-%m-%Y"))
    # plt.ylabel('Infectados')
    # plt.xlabel('semana epidemiológica')
    # plt.xlim((0,31))
    # plt.xticks(np.arange(1, 30+1, 1.0))
    # plt.legend()

plota_grafico_dia(dia,dataset_semanal)


# In[30]:


# Função para plotar a predição para um dia específico, COM REGRESSÃO.

dia='2020-03-8'

def plota_grafico_dia_regressao(dia,dataset_predicao):
    ''' Aceita dia como datetime ou dia no formato "%Y-%m-%d".
    '''

    if not isinstance(dia, datetime):
        dia=datetime.strptime(dia, "%Y-%m-%d")

    # Seleção dos dados da janela

    try:
        # 15 semanas antes.
        antes=dataset_predicao.loc[dia,['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15']]

        # 15 semanas preditas.
        depois=dataset_predicao.loc[dia,['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15']]

        # Monta a janela de visualização.
        janela=pd.concat([antes,depois])

        # Cria as séries de dados a partir de regressão dos dados da janela.

        serie_sigmoide_derivada1_janela=regressao_sigmoide_derivada1(janela)
        #serie_sigmoide_derivada1_depois=regressao_sigmoide_derivada1(depois)

        # Plotar dados

        plt.figure(figsize=(10,6))

        plt.bar(range(1, len(antes)+1),antes, label='Confirmed')
        plt.bar(range(len(antes)+1,len(antes)+len(depois)+1),depois,label='Predicted', color='navajowhite')

        # Plota regressão considerando todos os dados da janela
        plt.plot(range(1, len(janela)+1),serie_sigmoide_derivada1_janela,'--', color='orange')

        # Plota regressão considerando os dados preditos
        #plt.plot(range(len(antes)+1, len(janela)+1),serie_sigmoide_derivada1_depois,'--', color='orange')


        plt.title('Fortaleza: '+dia.strftime("%d-%m-%Y"))
        plt.ylabel('Infectados')
        plt.xlabel('semana epidemiológica')
        plt.xlim((0,31))
        plt.xticks(np.arange(1, 30+1, 1.0))
        plt.legend()
    except:
        print("Data não está na lista. Este erro pode ocorrer se as datas das amostras são semanais e a iteração está sendo diária.")

plota_grafico_dia_regressao(dia,dataset)


# <h4> Animação para visualização do modelo: dia após dia partindo de inicio_date </h4>

# In[ ]:


''' Animação para visualizar dados por semana epidemiológica (semana_epidemiologica) ou dia-a-dia (dataset).
'''


# Define qual série será mostrada:

#animation_data = dataset_semanal # Apresenta apenas as semanas epidemiológicas que iniciam no domingo.
animation_data = dataset # Apresenta todos os dias como início de semana epidemiológica.

# Define o limite para quantidade de infectados na visualização
max_num_confirmados=1000

# Define a lista de R2 da predição.
r2_mlnn_depois=[]
r2_mlnn_janela=[]

# Define o dia de início e fim da simulação.
inicio_date = "2010-02-10"#"2020-02-14"
fim_date = "2020-03-10"

# Converte para datetime.
dia = datetime.strptime(inicio_date, "%Y-%m-%d")
fim = datetime.strptime(fim_date, "%Y-%m-%d")


fig, ax = plt.subplots(1, figsize=(11,6))

fig.canvas.flush_events()

plt.ion()

fig.show()
fig.canvas.draw()

while dia <= fim:

    # Caso a data não esteja na lista de datas disponíveis para plotar: incrementa dia e passa para próxima iteração.
    if dia not in animation_data.index:
        dia = dia + timedelta(days=1)
        continue

    # Seleciona os dados do dia a plotar
    # Confirmados passado
    antes=animation_data.loc[dia,['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15']]

    # Preditos
    depois=animation_data.loc[dia,['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15']]

    # Confirmados futuro
    alvo=animation_data.loc[dia,['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']]

    # Montagem da janela de visualização.
    janela=pd.concat([antes,depois])

    # Define os ticks do gráfico
    #semana,ano = animation_data.loc[dia,['semana','ano']]

    # Cria as séries de dados a partir de regressão dos dados da janela.

    serie_sigmoide_derivada1_janela=regressao_sigmoide_derivada1(janela)
    serie_sigmoide_derivada1_depois=regressao_sigmoide_derivada1(depois)

    # Calcula r2 predição.
    r2_mlnn_depois.append(max(r2_score(np.array(serie_sigmoide_derivada1_depois),np.array(alvo)),0))

    r2_mlnn_janela.append(max(r2_score(np.array(serie_sigmoide_derivada1_janela),np.array(pd.concat([antes,alvo]))),0))

    # Define limites da área de plotagem.
    ax.clear()
    ax.set_ylim(0,max_num_confirmados)
    ax.set_xlim(0,len(janela)+1)

    # Plota gráfico de barras.
    ax.bar(range(1, len(antes)+1),antes, label='Confirmed', color='0.5')
    #ax.bar(range(len(antes)+1,len(antes)+len(depois)+1),depois,label='Predicted', color='navajowhite')

    ax.bar(range(len(antes)+1,len(antes)+len(depois)+1),alvo,label='to be confirmed', color='0.8') #color='dodgerblue'

    # Plota regressão considerando todos os dados da janela
    ax.plot(range(1, len(janela)+1),serie_sigmoide_derivada1_janela,':',label='Predicted', color='0') #color='orange'

    # Plota regressão considerando os dados preditos
    #ax.plot(range(len(antes)+1, len(janela)+1),serie_sigmoide_derivada1_depois,'--', color='red')

    # Preenche título e rótulos dos gráficos
    #ax.set_title('Infecteds in Fortaleza: '+ dia.strftime("%d-%m-%Y"))
    # ax.set_title('Infecteds in Fortaleza 2020')
    # ax.set_ylabel('Infecteds')
    # ax.set_xlabel('Epidemiological week')
    # ax.set_xticks(range(1, len(janela)+1))
    # ax.legend(loc="upper left")
    #
    # fig.canvas.draw()
    # plt.show(block=False)
    #plt.pause(0.001)

    dia = dia + timedelta(days=1)

plt.ioff()
