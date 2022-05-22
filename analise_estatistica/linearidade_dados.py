from array import array
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('../treino_teste\\Fortaleza_dataset_v3.csv',
                 sep=',',
                 encoding = 'utf8', parse_dates=['data_notifica'])

col_temp = ['dengue','dengue_sma7', 'acumulado_21','acumulado_mes','populacao','rt','susceptiveis','densidade_dem','precipitacao_sma7','temp_med_7sma','temp_min_7sma','umidade_7sma','vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

for i in range(len(col_temp)):
    colunas = col_temp + alvo
    x = data.iloc[:, 16:].values
    y = np.array(data[alvo[4]]).reshape(-1,1)
    x = np.array(x[:,i:i+1]).reshape(-1,1)

    xa = np.squeeze(x)
    ya = np.squeeze(y)
    slope, intercept, r_value, p_value, std_erro = stats.linregress(xa,ya)
    plt.plot(xa,ya, 'ro',color='#FF4500',alpha=0.5)
    plt.ylabel('5ª Semana')
    plt.xlabel(col_temp[i])
    # delimitando escala para "X" e "y". X de 0 a 600 e y de 0 a 5000
    #plt.axis([0, 40, 0, 1500])
    plt.plot(xa , xa * slope + intercept, 'b',color='blue')
    plt.show()
    print('#################################################################')
