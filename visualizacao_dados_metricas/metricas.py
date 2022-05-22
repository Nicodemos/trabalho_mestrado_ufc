import numpy as np
import matplotlib.pyplot as plt

# # Configuração
N = 5
valter = (49.91, 75.11, 90.87, 97.09, 109.52)
#valter = (49.91, 75.11, 90.87, 97.09, 109.52, 120.87, 129.82, 133.76)
#men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)
width = 0.15

fig, ax = plt.subplots()
rects1 = ax.bar(ind, valter, width, color='#8B0000')
#rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

mlp_nico = (35.96, 42.10, 50.78, 39.87, 41.92)
rects2 = ax.bar(ind+width, mlp_nico, width, color='#1C1C1C')

nicodemos = (34.91, 50.61, 45.44, 43.99, 36.68)
#nicodemos = (44.83, 65.12, 85.15, 98.24, 109.40, 120.90, 138.50, 162.11)
rects3 = ax.bar(ind+width*2, nicodemos, width, color='#BDB76B')

knn = (56.16, 69.06, 80.57, 87.44, 89.45)
#knn = (96.67, 132.78, 162.47, 191.28, 220.82, 239.88, 257.71, 273.94)
rects4 = ax.bar(ind+width*3, knn, width, color='#0000FF')

SVR = (60.09, 86.35, 104.69, 125.59, 143.55)
rects5 = ax.bar(ind+width*4, SVR, width, color='#FFD700')

LSTM = (37.54, 43.98, 45.90, 49.25, 46.24)
rects6 = ax.bar(ind+width*5, LSTM, width, color='#2F4F4F')

# add some text for labels, title and axes ticks
ax.set_ylabel('MAE',fontsize=12)
ax.set_title('MAE POR SEMANA')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('           Semana 01','           Semana 02','           Semana 03', '           Semana 04', '           Semana 05'),fontsize=12)
#ax.set_xticklabels(('Semana 01','Semana 02','Semana 03', 'Semana 04', 'Semana 05','Semana 06','Semana 07','Semana 08'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('MLP/VALTER','MLP/NICODEMOS', 'XGboost', 'KNN', 'SVR', 'LSTM'),fontsize=12)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, 1.05*height,
                '%d' % int(height),
                ha='center', va='center',fontsize=12)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
plt.show()

###################################################################################################

N = 5
valter = (97.13, 93.15, 89.00, 83.04 , 77.84)
#valter = (97.13, 93.15, 89.00, 83.04 , 77.84, 75.04, 75.54, 74.85)
#men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.15      # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, valter, width, color='#8B0000')
#rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

mlp_nico = (98.82, 97.39, 96.52, 98.00, 98.41)
rects2 = ax.bar(ind+width, mlp_nico, width, color='#1C1C1C')

nicodemos = (98.63, 93.24, 94.47, 96.14, 98.74)
#nicodemos = (98.02, 95.10, 91.76 , 88.27, 86.21, 84.88, 79.68, 74.71)
#women_std = (3, 5, 2, 3, 3)
rects3 = ax.bar(ind+width*2, nicodemos, width, color='#BDB76B')
#rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

#knn = (86.27 , 76.15, 67.03 , 56.02, 46.03, 40.11, 35.84, 32.57)
knn = (95.69, 90.69, 86.89, 87.30, 88.74)
#women_std = (3, 5, 2, 3, 3)
rects4 = ax.bar(ind+width*3, knn, width, color='#0000FF')

SVR = (97.15, 91.06, 82.96, 73.19, 58.43)
#women_std = (3, 5, 2, 3, 3)
rects5 = ax.bar(ind+width*4, SVR, width, color='#FFD700')

LSTM = (98.69, 97.46, 98.11, 97.12, 97.87)
#women_std = (3, 5, 2, 3, 3)
rects6 = ax.bar(ind+width*5, LSTM, width, color='#2F4F4F',fontsize=12)

# add some text for labels, title and axes ticks
ax.set_ylabel('R2',fontsize=12)
ax.set_title('R2 POR SEMANA')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('           Semana 01','           Semana 02','           Semana 03', '           Semana 04', '           Semana 05'),fontsize=12)
#ax.set_xticklabels(('Semana 01','Semana 02','Semana 03', 'Semana 04', 'Semana 05','Semana 06','Semana 07','Semana 08'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('MLP/VALTER','MLP/NICODEMOS', 'XGboost', 'KNN', 'SVR', 'LSTM'),fontsize=12)

def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, 1.05*height,
                '%d' % int(height),
                ha='center', va='center')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
plt.show()
