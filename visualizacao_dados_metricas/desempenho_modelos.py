import numpy as np
import matplotlib.pyplot as plt
# # Configuração
N = 5

# métrica r²
valter = np.array([97.13, 93.15, 89.00, 83.04 , 77.84])
mlp_nico = np.array([98.82, 97.39, 96.52, 98.00, 98.41])
nicodemos = np.array([98.63, 93.24, 94.47, 96.14, 98.74])
knn = np.array([95.69, 90.69, 86.89, 87.30, 88.74])
SVR = np.array([97.15, 91.06, 82.96, 73.19, 58.43])
LSTM = np.array([98.69, 97.46, 97.87, 96.85, 96.90])

# # métrica mae²
# valter = np.array([49.91, 75.11, 90.87, 97.09, 109.52])
# mlp_nico = np.array([35.96, 42.10, 50.78, 39.87, 41.92])
# nicodemos = np.array([34.91, 50.61, 45.44, 43.99, 36.68])
# knn = np.array([56.16, 69.06, 80.57, 87.44, 89.45])
# SVR = np.array([60.09, 86.35, 104.69, 125.59, 143.55])
# LSTM = np.array([37.54, 43.98, 45.90, 51.72, 53.48])

x = np.array(['SEMANA_01', 'SEMANA_02', 'SEMANA_03', 'SEMANA_04', 'SEMANA_05'])

plt.errorbar(x,valter,label='MLP/Valter',marker='s')
plt.errorbar(x,mlp_nico,label='MLP',marker='*')
plt.errorbar(x,nicodemos,label='XGBoost',marker='o')
plt.errorbar(x,knn,label='KNN',marker='v')
plt.errorbar(x,SVR,label='SVR',marker='<')
plt.errorbar(x,LSTM,label='LSTM',marker='+')
plt.legend(loc='lower left',fontsize=20)
#plt.legend(loc='upper left',fontsize=20)
plt.show()
