import scipy.stats as sh
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('../treino_teste\\Fortaleza_dataset_v3.csv',
                 sep=',',
                 encoding = 'utf8', parse_dates=['data_notifica'])

col_temp = ['Dengue','Dengue_sma7', 'Acumulado_21','Acumulado_mes','Populacao','RT','Susceptiveis','Sensidade_dem','Precipitacao_sma7','Temp_med_7sma','Temp_min_7sma','Umidade_7sma','Vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

def kolmogorov_smirnov_critico(n):
    # table of critical values for the kolmogorov-smirnov test - 95% confidence
    # Source: https://www.soest.hawaii.edu/GG/FACULTY/ITO/GG413/K_S_Table_one_Sample.pdf
    # Source: http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    # alpha = 0.05 (95% confidential level)

    if n <= 40:
        # valores entre 1 e 40
        kolmogorov_critico = [0.97500, 0.84189, 0.70760, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925,
                      0.39122, 0.37543, 0.36143, 0.34890, 0.33760, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408,
                      0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25907, 0.25438, 0.24993, 0.24571, 0.24170,
                      0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012]
        ks_critico = kolmogorov_critico[n - 1]
        return ks_critico

    elif n > 40:
        # valores acima de 40:
        kolmogorov_critico = 1.36/(np.sqrt(n))
        ks_critico = kolmogorov_critico
        return ks_critico
plt.figure(figsize=(5,3))
colunas = col_temp+alvo
x = data.iloc[:,16:].values
for i in range(len(col_temp)):

    y = np.squeeze(x[:,i:i+1])
    media = y.mean()
    print('media: ',media)
    std = np.std(y)
    print('desvio padrão: ',std)
    ks_stat, valor_p = sh.kstest(y,cdf='norm', args=(media, std), N = len(y))

    print('Teste de Kolmogorov-Smirnov para a Variável ',col_temp[i])
    print('Statistica: ',ks_stat)
    print('P_value: ',valor_p)


    ks_critico = kolmogorov_smirnov_critico(len(y))
    print('valor critico: ',ks_critico)
    print('P_value: ',ks_critico)
    if ks_critico >= ks_stat:
        print("Com 95% de confianca, não temos evidências para rejeitar a hipótese de normalidade dos dados, segundo o teste de Kolmogorov-Smirnov")
    else:
        print("Com 95% de confianca, temos evidências para rejeitar a hipótese de normalidade dos dados, segundo o teste de Kolmogorov-Smirnov")

    df = pd.DataFrame(x[:,i:i+1],columns=[col_temp[i]])
    df.sort_values([col_temp[i]])
    print('------------------------------------------------------------------------------------------------------')

    df.plot(kind='hist',color="Blue", alpha=0.5,bins=30,figsize=(6,3))

    #plt.show()

    #print('#################################################################')

