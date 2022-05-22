import scipy.stats as sh
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('../treino_teste\\Fortaleza_dataset_v3.csv',
                 sep=',',
                 encoding = 'utf8', parse_dates=['data_notifica'])

col_temp = ['dengue','dengue_sma7', 'acumulado_21','acumulado_mes','populacao','rt','susceptiveis','densidade_dem','precipitacao_sma7','temp_med_7sma','temp_min_7sma','umidade_7sma','vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

colunas = col_temp+alvo
x = data.iloc[:,16:].values
for i in range(len(col_temp)):
    shap_stat, valor_p = sh.shapiro(np.sort(np.squeeze(x[:,i:i+1])))
    print('Teste de Shapiro Wilk para a Variável ',col_temp[i])
    print('Statistica: ',shap_stat)
    print('P_value: ',valor_p)

    df = pd.DataFrame(x[:,i:i+1],columns=[col_temp[i]])
    df.sort_values([col_temp[i]])
    df.plot(kind='hist')
    plt.show()
    print('#################################################################')







