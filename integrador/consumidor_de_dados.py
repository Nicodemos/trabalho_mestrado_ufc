import requests as req
from flask import Flask
import modelos_para_uso.inferencia as inf

app = Flask(__name__)
#inf = modelos_para_uso.inferencia()

# para evitar divisão por zero
def validacao(dado, contador):
    if dado == 0:
        return 0
    elif contador == 0:
        return dado
    else:
        return dado/contador

## calcula toda média movel dos atributos
def extrai_dados_json(variavel):
     d_dengue = 0.00
     d_dengue_sma7 = 0.00
     d_acum21 = 0.00
     d_densidade = 0.00
     d_populacao = 0.00
     d_chava = 0.00
     d_humd = 0.00
     d_vento = 0.00
     d_acumulado =0.00
     d_date = None

     for dado in variavel['Dengue']:
         if dado['value'] != None:
            d_dengue = float(dado['value'])

     for dado in variavel['Dengue_sma7']:
         if dado['value'] != None:
            d_dengue_sma7 = float(dado['value'])

     for dado in variavel['Acumulado_21']:
         if dado['value'] != None:
            d_acum21 = float(dado['value'])

     for dado in variavel['Acumulado']:
         if dado['value'] != None:
            d_acumulado = float(dado['value'])

     for dado in variavel['Densidade_Dem']:
         if dado['value'] != None:
            d_densidade = float(dado['value'])

     for dado in variavel['Populacao']:
          if dado['value'] != None:
             d_populacao = int(dado['value'])

     for dado in variavel['data_notifica']:
         if dado['value'] != None:
            d_date = dado['value']

     for dado in variavel['Precipitacao_sma7']:
         if dado['value'] != None:
            d_chava = float(dado['value'])

     for dado in variavel['Vento_mps_7sma']:
         if dado['value'] != None:
            d_vento = float(dado['value'])

     for dado in variavel['Umidade_7sma']:
         if dado['value'] != None:
           d_humd = float(dado['value'])

     list_diario = []
     list_diario.append(d_date)
     list_diario.append(d_dengue)
     list_diario.append(d_dengue_sma7)
     list_diario.append(d_acum21)
     list_diario.append(d_acumulado)
     list_diario.append(d_populacao)
     list_diario.append(d_densidade)
     list_diario.append(d_chava)
     list_diario.append(d_humd)
     list_diario.append(d_vento)

     return list_diario

@app.route('/calc')
def get_dojot():
    ## se for enviado a sma para a dojot nos dados horarios, recuperar os 7 ultimos envio
    ## caso contrário, se enviar 24 dados horario, dados brutos, recuprar 168
    #url_dados_horario = "http://192.168.1.8:8000/history/device/ad9251/history?lastN={}".format(168)
    url_dados_= "http://192.168.1.16:8000/history/device/a76641/history?lastN={}".format(1)
    param = {'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJXeEs5UmE5MHlwTzBrZTFDMlhJSmZUOHd0aEFUd24xMiIsImlhdCI6MTYxMDUwMzQ0NiwiZXhwIjoxNjEwNTAzODY2LCJwcm9maWxlIjoiYWRtaW4iLCJncm91cHMiOlsxXSwidXNlcmlkIjoxLCJqdGkiOiIyZjkxMjRkNjkyYWYzNDA4ZjViYTdiOWNhZTdhMGM2NCIsInNlcnZpY2UiOiJhZG1pbiIsInVzZXJuYW1lIjoiYWRtaW4ifQ.LHapfM-SyfYG8WqBTYcgshsV3D9O-9QjqfHfSTJN4FE'}

    resp_diarios = req.get(url_dados_, headers=param)
    if resp_diarios.status_code != 200:
        print('Resposta: ',resp_diarios.status_code)
    else:
        content_diario = resp_diarios.json()
        dados = extrai_dados_json(content_diario)
        print(inf.inferencia(nome='xgboost',tipo=None).go_ml(dados[1:]))
        resultado_inferencia = inf.inferencia(nome='xgboost',tipo=None).go_ml(extrai_dados_json(content_diario)[1:])
        resultado_inferencia = str(resultado_inferencia[0]).split('.')[0]

        return str("PREDIÇÃO PARA O DIA {}".format(dados[:1][0].split()[0])+" = "+resultado_inferencia)

app.run(host='192.168.1.14',port=5001,debug=True)
