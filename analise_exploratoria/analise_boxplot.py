import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('../treino_teste/Fortaleza_dataset_v3.csv',
                 sep=',',
                 encoding = 'utf8', parse_dates=['data_notifica'])

col_temp = ['data_notifica','Dengue','Dengue_sma7', 'Acumulado_21','Acumulado','Populacao','Rt','Susceptiveis','Densidade_Dem','Precipitacao_sma7','Temp_med_7sma','Temp_min_7sma','Umidade_7sma','Vento_mps_7sma']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

def gera_boxplot_graf(xx, df, msg):
    xx.boxplot(df[msg],notch=True,vert=True,patch_artist=True)
    xx.set_title('{}'.format(msg))

fig01, (ax) = plt.subplots(nrows=3,ncols=6)
data_graf = pd.DataFrame(data, columns=col_temp+alvo)

i=0
j=0
for x in range(1,14):
    gera_boxplot_graf(ax[i][j],data_graf,col_temp[x])
    if x==6:
        i+=1
        j=-1
    if x==12:
        i+=1
        j=-1
    j+=1
plt.tight_layout()
plt.show()

