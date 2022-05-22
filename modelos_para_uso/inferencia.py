import pickle as pk
from sklearn.preprocessing import StandardScaler
import numpy as np
class inferencia:
    def __init__(self,nome,tipo):
        self.tipo = tipo
        self.padronizador = StandardScaler()
        self.nome_modelo = nome+'.pkl'
        self.nome_padronizador_y = 'xgboost_normalizay.pkl'
        with open('../modelos_para_uso/'+self.nome_modelo, 'rb') as f:
            self.modelo = pk.load(f)

        with open('../modelos_para_uso/'+self.nome_padronizador_y, 'rb') as f:
            self.padronizador_y = pk.load(f)

    def go_ml(self,dados):
        _array = np.array(dados).reshape(1,-1)
        #dados_padronizados = self.padronizador.fit_transform(_array.reshape(1,-1))
        print("shape: ",np.shape(_array))
        resultado = self.modelo.predict(_array)
        return resultado
