import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)
data=pd.read_csv('../treino_teste\\Fortaleza_dataset_v3.csv',
                 sep=',',
                 encoding = 'utf8', parse_dates=['data_notifica'])

col_temp = ['Dengue','Deng_7', 'Deng_21','Acumulado','Populacao','Reproduti','Suscepti','Dens_dem','Prec_7','Temp_med_7','Temp_min_7','Umidade_7','Vent_vel_7']
alvo=['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15']

colunas = col_temp+alvo
x = data.iloc[:,16:].values

df = pd.DataFrame(x,columns=colunas)

n_alvos = len(alvo)

correlations = df.corr(method = 'spearman')
correlations_p = df.corr(method = 'pearson')
for alvo in df.columns[-n_alvos:]:
    print('ALVO: ', df.columns[-n_alvos:])
    print('corr: ', correlations[alvo])
    plt.figure(figsize=(10,6))

    spearman = correlations[alvo].drop(df.columns[-n_alvos:]).sort_values(ascending=False)
    pearson = correlations_p[alvo].drop(df.columns[-n_alvos:]).sort_values(ascending=False)

    plot_graf = pd.DataFrame(columns=['Spearman','Pearson'])
    plot_graf['Spearman'] = spearman
    plot_graf['Pearson'] =  pearson
    plot_graf['Spearman'].plot(kind='barh',color="0.8")
    plot_graf['Pearson'].plot(kind='barh',color="#4F4F4F")
    plt.xlim(-1,1)
    plt.xlabel('Correlação', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(None)
    plt.legend(loc='upper right',fontsize=16)
    plt.show()





