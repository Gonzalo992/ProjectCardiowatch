import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash
import requests
import statsmodels.api as sm
import itertools
import math
import dash_bootstrap_components as dbc
import csv

app = Dash(__name__)


######### Carga de dataset #########
df = pd.read_csv("https://raw.githubusercontent.com/Gonzalo992/dataset/main/Datos_definitivo.csv",sep=',',encoding='latin-1')

######### Carga de geojson #########
repo_url = 'https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson'
pe_distrito_geo = requests.get(repo_url).json()

######### Filtrar geojson provincia LIMA #########
filtered_features = []
for feature in pe_distrito_geo['features']:
    if feature['properties']['NOMBPROV'] == 'LIMA':
        filtered_features.append(feature)
filtered_data = {'type': 'FeatureCollection', 'features': filtered_features}

######### Preparación de Dataset #########
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')

######### Components HTML #########
header = navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Dropdown(id="slct_year",
                            options=[
                                {"label": "2014", "value": 2014},
                                {"label": "2015", "value": 2015},
                                {"label": "2016", "value": 2016},
                                {"label": "2017", "value": 2017},
                                {"label": "2018", "value": 2018},
                                {"label": "2019", "value": 2019},
                                {"label": "2020", "value": 2020},
                                {"label": "2021", "value": 2021},
                                {"label": "2022", "value": 2022},
                                {"label": "2023", "value": 2023},
                                {"label": "2024", "value": 2024},
                                {"label": "2025", "value": 2025}],
                                multi=False,
                                value=2019,
                                style = dict(
                                width = '120px',
                                margin = '0px 10px 0px 0px',
                                )
            )
        ),
        dbc.NavItem(
            dcc.Dropdown(id="slct_disease",
                    options=[
                        {"label": "Seleccionar todo", "value": 'todo'},
                        {"label": "Hipertensión esencial (Primaria)", "value": 'HIPERTENSION ESENCIAL (PRIMARIA)'},
                        {"label": "Hipertensión secundaria", "value": 'HIPERTENSION SECUNDARIA'},
                        {"label": "Otras arritmias cardiacas", "value": 'OTRAS ARRITMIAS CARDIACAS'},
                        {"label": "Venas varicosas de los miembros inferiores", "value": 'VENAS VARICOSAS DE LOS MIEMBROS INFERIORES'}],
                    multi=False,
                    value='todo',
                    style = dict(
                                width = '320px',
                                margin = '0px 10px 0px 0px',
                            )
            )
        ),
        dbc.NavItem(
            dcc.Dropdown(id="slct_distrito",
                    options=[
                        {"label": "Seleccionar todo", "value": 'todo'},
                        {"label": "Ancón", "value": 'ANCON'},
                        {"label": "ate", "value": 'ATE'},
                        {"label": "Barranco", "value": 'BARRANCO'},
                        {"label": "Breña", "value": 'BREÑA'},
                        {"label": "Carabayllo", "value": 'CARABAYLLO'},
                        {"label": "Chaclacayo", "value": 'CHACLACAYO'},
                        {"label": "Chorrillos", "value": 'CHORRILLOS'},
                        {"label": "Cieneguilla", "value": 'CIENEGUILLA'},
                        {"label": "Comas", "value": 'COMAS'},
                        {"label": "El agustino", "value": 'EL AGUSTINO'},
                        {"label": "Independencia", "value": 'INDEPENDENCIA'},
                        {"label": "Jesus maria", "value": 'JESUS MARIA'},
                        {"label": "La molina", "value": 'LA MOLINA'},
                        {"label": "La victoria", "value": 'LA VICTORIA'},
                        {"label": "Lima", "value": 'LIMA'},
                        {"label": "Lince", "value": 'LINCE'},
                        {"label": "Los olivos", "value": 'LOS OLIVOS'},
                        {"label": "Lurigancho", "value": 'LURIGANCHO'},
                        {"label": "Lurin", "value": 'LURIN'},
                        {"label": "Magdalena del mar", "value": 'MAGDALENA DEL MAR'},
                        {"label": "Miraflores", "value": 'MIRAFLORES'},
                        {"label": "Pachacamac", "value": 'PACHACAMAC'},
                        {"label": "Pucusana", "value": 'PUCUSANA'},
                        {"label": "Pueblo libre", "value": 'PUEBLO LIBRE'},
                        {"label": "Puente piedra", "value": 'PUENTE PIEDRA'},
                        {"label": "Punta hermosa", "value": 'PUNTA HERMOSA'},
                        {"label": "Punta negra", "value": 'PUNTA NEGRA'},
                        {"label": "Rimac", "value": 'RIMAC'},
                        {"label": "San bartolo", "value": 'SAN BARTOLO'},
                        {"label": "San borja", "value": 'SAN BORJA'},
                        {"label": "San isidro", "value": 'SAN ISIDRO'},
                        {"label": "San juan de lurigancho", "value": 'SAN JUAN DE LURIGANCHO'},
                        {"label": "San juan de miraflores", "value": 'SAN JUAN DE MIRAFLORES'},
                        {"label": "San luis", "value": 'SAN LUIS'},
                        {"label": "San martin de porres", "value": 'SAN MARTIN DE PORRES'},
                        {"label": "San miguel", "value": 'SAN MIGUEL'},
                        {"label": "Santa anita", "value": 'SANTA ANITA'},
                        {"label": "Santa maria del mar", "value": 'SANTA MARIA DEL MAR'},
                        {"label": "Santa rosa", "value": 'SANTA ROSA'},
                        {"label": "Santiago de surco", "value": 'SANTIAGO DE SURCO'},
                        {"label": "Surquillo", "value": 'SURQUILLO'},
                        {"label": "Villa el salvador", "value": 'VILLA EL SALVADOR'},
                        {"label": "Villa maria del triunfo", "value": 'VILLA MARIA DEL TRIUNFO'},
                        {"label": "Villa maria del triunfo", "value": 'VILLA MARIA DEL TRIUNFO'}],
                    multi=False,
                    value='todo',
                    style = dict(
                                width = '120px',
                                )
            )
        ),
        dbc.NavItem(
            dbc.Button("Analizar", color="light", className="me-1", id='submit-button-state', n_clicks=0)
        )
    ],
    brand="CardioWatch",
    brand_href="#",
    color="dark",
    dark=True,
    )

card_content = [
    dbc.CardBody(
        [
            html.H5("Total de enfermos", className="card-title"),
            html.P(
                className="card-text",id="output_container"
            ),
        ]
    ),
]

card_content2 = [
    dbc.CardBody(
        [
            html.H5("Distrito mayor enfermos", className="card-title"),
            html.P(
                className="card-text",id="output_container_max_distrito",
            ),
        ]
    ),
]

cards = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(card_content, color="dark", inverse=True)),
                dbc.Col(dbc.Card(card_content2, color="dark", inverse=True), className="card2"),
            ],className='cardline'
        )
    ],style={"width": "100%","margin":"0px","display":"flex"},className="container carddata"
)

######### Función para obtener PDQ óptimos #########
def getPDQ(datos):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    dic_aic={}
    list_param=[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            list_param=[]
            try:
                mod = sm.tsa.statespace.SARIMAX(datos, order = param, seasonal_order = param_seasonal, enforce_stationary = False,enforce_invertibility=False)
                result = mod.fit()
                list_param.append(param)
                list_param.append(param_seasonal)
                dic_aic[result.aic] = list_param
            except:
                continue
    dic_value=dic_aic[min(dic_aic.keys())]
    return dic_value

######### Función modelo SARIMAX #########
def modelo_SARIMAX(datos,time_predict,steps_predict,predict,value,value_seasonal):
    uc_ci=[]
    model = sm.tsa.statespace.SARIMAX(datos, order = value,
                                      seasonal_order = value_seasonal
                                     )
    result = model.fit()

    prediction = result.get_prediction(start = pd.to_datetime(time_predict), dynamic = False)
    prediction_ci = prediction.conf_int()
    prediction_ci

    pred_uc = result.get_forecast(steps = steps_predict)
    pred_ci = pred_uc.conf_int()
    uc_ci.append(pred_uc)
    uc_ci.append(pred_ci)

    return uc_ci

######### Función modelo SARIMAX para datos pasados #########
def modelo_SARIMAX_past(datos,value,value_seasonal):
    model = sm.tsa.statespace.SARIMAX(datos, order = value,seasonal_order = value_seasonal)
    result = model.fit()
    return result

######### Función convertir lista a string #########
def convertString(list):
    res = str("".join(map(str, list)))
    return res

######### Variables color sexo, año, enfermedad y distrito #########
colors = ['#DF01D7', '#084B8A']
lblAnio=""
lblEnfermedad=""
lblDistrito=""

######### Función para actualizar gráficos #########
def update_graph(n_clicks,option_slctd,option2_slctd,option3_slctd):
    lblAnio=option_slctd
    lblEnfermedad=option2_slctd
    lblDistrito=option3_slctd

    lDistritos=['ATE','ANCON','BARRANCO','BREÑA','CARABAYLLO','CHACLACAYO','CHORRILLOS','CIENEGUILLA','COMAS','EL AGUSTINO','INDEPENDENCIA','JESUS MARIA','LA MOLINA','LA VICTORIA','LIMA','LINCE','LOS OLIVOS','LURIGANCHO','LURIN','MAGDALENA DEL MAR','MIRAFLORES','PACHACAMAC','PUCUSANA','PUEBLO LIBRE','PUENTE PIEDRA','PUNTA HERMOSA','PUNTA NEGRA','RIMAC','SAN BARTOLO','SAN BORJA','SAN ISIDRO','SAN JUAN DE LURIGANCHO','SAN JUAN DE MIRAFLORES','SAN LUIS','SAN MARTIN DE PORRES','SAN MIGUEL','SANTA MARIA DEL MAR','SANTA ROSA','SANTIAGO DE SURCO','SURQUILLO','VILLA EL SALVADOR','VILLA MARIA DEL TRIUNFO']
    lEnfermedades=['HIPERTENSION ESENCIAL (PRIMARIA)','HIPERTENSION SECUNDARIA','OTRAS ARRITMIAS CARDIACAS','VENAS VARICOSAS DE LOS MIEMBROS INFERIORES']
    dfLine = df.copy()
    dfPieMasculino = df.copy()
    dfPieFemenino = df.copy()
    dfChoropleth = df.copy()
    dfLineAll = df.copy()

    dff = df.copy()
    dff2 = df.copy()
    dff3 = df.copy()
    dff4 = df.copy()

    fig=""
    fig2=""
    fig3=""
    container=""
    container_max_distrito=""
    indice_maximo=""
    msarimax=[]
    steps=0
    dfModelos = pd.read_csv('https://raw.githubusercontent.com/Gonzalo992/dataset/main/modelos.csv',sep=',',encoding='latin-1')
    if(option_slctd>=2023):
        if(option2_slctd=='todo' and option3_slctd=='todo'):
            dfLine = dfLine.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLine = dfLine.set_index('Fecha')
            dfLine = dfLine.asfreq('MS')
            dfLine = dfLine.sort_index()

            value=getPDQ(dfLine)[0]
            value_seasonal=getPDQ(dfLine)[1]

            if(option_slctd==2023):
                steps=12
            elif(option_slctd==2024):
                steps=24
            elif(option_slctd==2025):
                steps=36

            msarimax=modelo_SARIMAX(dfLine,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            pred_data=pd.DataFrame(msarimax[0].predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']

            pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))
            fig = px.line(pred_data.tail(n=12), x='Fecha', y='Pronosticos',markers=True,template='seaborn')
            fig.update_layout(title={
            'text': 'Total de casos de enfermedades cardiovasculares<br>'+"en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

            container = int(pred_data.tail(n=12)['Pronosticos'].sum())

            dfPieMasculino=  dfPieMasculino[dfPieMasculino['Sexo']=='Masculino']
            dfPieFemenino=  dfPieFemenino[dfPieFemenino['Sexo']=='Femenino']
            dfPieMasculino = dfPieMasculino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfPieFemenino = dfPieFemenino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfPieMasculino = dfPieMasculino.set_index('Fecha')
            dfPieMasculino = dfPieMasculino.asfreq('MS')
            dfPieMasculino = dfPieMasculino.sort_index()

            dfPieFemenino = dfPieFemenino.set_index('Fecha')
            dfPieFemenino = dfPieFemenino.asfreq('MS')
            dfPieFemenino = dfPieFemenino.sort_index()

            msarimaxMas=modelo_SARIMAX(dfPieMasculino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            msarimaxFem=modelo_SARIMAX(dfPieFemenino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

            pred_dataMas=pd.DataFrame(msarimaxMas[0].predicted_mean)
            pred_dataMas = pred_dataMas.reset_index()
            pred_dataMas.columns=['Fecha','Pronosticos']

            pred_dataFem=pd.DataFrame(msarimaxFem[0].predicted_mean)
            pred_dataFem = pred_dataFem.reset_index()
            pred_dataFem.columns=['Fecha','Pronosticos']

            pred_dataMas.loc[pred_dataMas["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_dataFem.loc[pred_dataFem["Pronosticos"] < 0, "Pronosticos"] = 0

            fig2 = px.pie(values=[pred_dataMas.tail(n=12)['Pronosticos'].sum(),pred_dataFem.tail(n=12)['Pronosticos'].sum()], names=['Masculino','Femenino'], title='Porcentaje de casos pronosticados de enfermedades cardiovasculares<br>por sexo en el año '+str(lblAnio)+' en todos los distritos de Lima',color_discrete_sequence=colors,template='seaborn')

            fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })
            fig2.update_traces(hovertemplate = None,
                  hoverinfo = "skip")

            dicdataCho={}
            ldataCho=[]

            for i in lDistritos:
                dfChoropleth = df.copy()
                ldataCho=[]
                dfChoropleth=  dfChoropleth[dfChoropleth['Distrito']==i]
                dfChoropleth = dfChoropleth.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
                dfChoropleth = dfChoropleth.set_index('Fecha')
                dfChoropleth = dfChoropleth.asfreq('MS')
                dfChoropleth = dfChoropleth.sort_index()

                msarimaxCho=modelo_SARIMAX(dfChoropleth,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

                pred_dataCho=pd.DataFrame(msarimaxCho[0].predicted_mean)
                pred_dataCho = pred_dataCho.reset_index()
                pred_dataCho.columns=['Fecha','Pronosticos']

                ldataCho.append(pred_dataCho.tail(n=12)['Pronosticos'].sum())
                dicdataCho[i] = ldataCho

            pddatacho = pd.DataFrame.from_dict(dicdataCho)
            pddatachomel = pd.melt(pddatacho, id_vars=None, value_vars=None)

            fig3 = px.choropleth_mapbox(data_frame=pddatachomel,
                geojson=filtered_data,
                locations='variable',
                featureidkey='properties.NOMBDIST',
                color='value',
                color_continuous_scale="oryel",
                mapbox_style="open-street-map",
                center={"lat": -12.04318, "lon": -77.02824},
                zoom=8,
                opacity=0.7,
            )
            fig3.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")
            fig3.update_layout(title={
            'text':'Mapa de incidencias cardiovasculares',
            'font':{'family': 'Arial', 'size': 16},
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            })
            indice_maximo = pddatachomel['value'].idxmax()
            container_max_distrito = pddatachomel.loc[indice_maximo, 'variable']

        elif(option2_slctd=='todo' and option3_slctd!='todo'):
            dfLine = dfLine[dfLine["Categoria_enfermedad"].isin(lEnfermedades)]
            dfLine = dfLine[dfLine["Distrito"] == option3_slctd]
            dfLine = dfLine.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLine = dfLine.set_index('Fecha')
            dfLine = dfLine.asfreq('MS')
            dfLine = dfLine.sort_index()

            value=getPDQ(dfLine)[0]
            value_seasonal=getPDQ(dfLine)[1]

            if(option_slctd==2023):
                steps=12
            elif(option_slctd==2024):
                steps=24
            elif(option_slctd==2025):
                steps=36

            msarimax=modelo_SARIMAX(dfLine,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            pred_data=pd.DataFrame(msarimax[0].predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']

            pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))
            fig = px.line(pred_data.tail(n=12), x='Fecha', y='Pronosticos',markers=True,template='seaborn')
            fig.update_layout(title={
            'text': "Total de casos de enfermedades cardiovasculares<br>en el año "+str(lblAnio)+" en el distrito de "+str(lblDistrito).lower(),
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

            container = int(pred_data.tail(n=12)['Pronosticos'].sum())



            dfPieMasculino = dfPieMasculino[dfPieMasculino["Categoria_enfermedad"].isin(lEnfermedades)]
            dfPieMasculino = dfPieMasculino[dfPieMasculino["Distrito"] == option3_slctd]

            dfPieFemenino = dfPieFemenino[dfPieFemenino["Categoria_enfermedad"].isin(lEnfermedades)]
            dfPieFemenino = dfPieFemenino[dfPieFemenino["Distrito"] == option3_slctd]

            dfPieMasculino=  dfPieMasculino[dfPieMasculino['Sexo']=='Masculino']
            dfPieFemenino=  dfPieFemenino[dfPieFemenino['Sexo']=='Femenino']
            dfPieMasculino = dfPieMasculino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfPieFemenino = dfPieFemenino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfPieMasculino = dfPieMasculino.set_index('Fecha')
            dfPieMasculino = dfPieMasculino.asfreq('MS')
            dfPieMasculino = dfPieMasculino.sort_index()

            dfPieFemenino = dfPieFemenino.set_index('Fecha')
            dfPieFemenino = dfPieFemenino.asfreq('MS')
            dfPieFemenino = dfPieFemenino.sort_index()

            msarimaxMas=modelo_SARIMAX(dfPieMasculino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            msarimaxFem=modelo_SARIMAX(dfPieFemenino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

            pred_dataMas=pd.DataFrame(msarimaxMas[0].predicted_mean)
            pred_dataMas = pred_dataMas.reset_index()
            pred_dataMas.columns=['Fecha','Pronosticos']

            pred_dataFem=pd.DataFrame(msarimaxFem[0].predicted_mean)
            pred_dataFem = pred_dataFem.reset_index()
            pred_dataFem.columns=['Fecha','Pronosticos']

            pred_dataMas.loc[pred_dataMas["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_dataFem.loc[pred_dataFem["Pronosticos"] < 0, "Pronosticos"] = 0

            fig2 = px.pie(values=[pred_dataMas.tail(n=12)['Pronosticos'].sum(),pred_dataFem.tail(n=12)['Pronosticos'].sum()], names=['Masculino','Femenino'], title='Porcentaje de casos pronosticados de enfermedades cardiovasculares<br>por sexo en el año '+str(lblAnio)+' en '+str(lblDistrito).lower(),color_discrete_sequence=colors,template='seaborn')

            fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })
            fig2.update_traces(hovertemplate = None,
                  hoverinfo = "skip")
            dicdataCho={}
            ldataCho=[]
            lDistritos=[option3_slctd]
            for i in lDistritos:
                dfChoropleth = df.copy()
                ldataCho=[]
                dfChoropleth = dfChoropleth[dfChoropleth["Categoria_enfermedad"].isin(lEnfermedades)]
                dfChoropleth=  dfChoropleth[dfChoropleth['Distrito']==i]
                dfChoropleth = dfChoropleth.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
                dfChoropleth = dfChoropleth.set_index('Fecha')
                dfChoropleth = dfChoropleth.asfreq('MS')
                dfChoropleth = dfChoropleth.sort_index()

                msarimaxCho=modelo_SARIMAX(dfChoropleth,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

                pred_dataCho=pd.DataFrame(msarimaxCho[0].predicted_mean)
                pred_dataCho = pred_dataCho.reset_index()
                pred_dataCho.columns=['Fecha','Pronosticos']

                ldataCho.append(pred_dataCho.tail(n=12)['Pronosticos'].sum())
                dicdataCho[i] = ldataCho

            pddatacho = pd.DataFrame.from_dict(dicdataCho)
            pddatachomel = pd.melt(pddatacho, id_vars=None, value_vars=None)

            fig3 = px.choropleth_mapbox(data_frame=pddatachomel,
                geojson=filtered_data,
                locations='variable',
                featureidkey='properties.NOMBDIST',
                color='value',
                color_continuous_scale="oryel",
                mapbox_style="open-street-map",
                center={"lat": -12.04318, "lon": -77.02824},
                zoom=8,
                opacity=0.7,
            )
            fig3.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")
            fig3.update_layout(title={
            'text':'Mapa de incidencias cardiovasculares',
            'font':{'family': 'Arial', 'size': 16},
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            })
            indice_maximo = pddatachomel['value'].idxmax()
            container_max_distrito = pddatachomel.loc[indice_maximo, 'variable']

        elif(option2_slctd!='todo' and option3_slctd=='todo'):
            dfLine = dfLine[dfLine["Categoria_enfermedad"] == option2_slctd]
            dfLine = dfLine[dfLine["Distrito"].isin(lDistritos)]

            dfLine = dfLine.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLine = dfLine.set_index('Fecha')
            dfLine = dfLine.asfreq('MS')
            dfLine = dfLine.sort_index()

            value=getPDQ(dfLine)[0]
            value_seasonal=getPDQ(dfLine)[1]

            if(option_slctd==2023):
                steps=12
            elif(option_slctd==2024):
                steps=24
            elif(option_slctd==2025):
                steps=36

            msarimax=modelo_SARIMAX(dfLine,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            pred_data=pd.DataFrame(msarimax[0].predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']

            pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))
            fig = px.line(pred_data.tail(n=12), x='Fecha', y='Pronosticos',markers=True,template='seaborn')
            fig.update_layout(title={
            'text': 'Nro. de casos pronosticados con '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

            container = int(pred_data.tail(n=12)['Pronosticos'].sum())



            dfPieMasculino = dfPieMasculino[dfPieMasculino["Categoria_enfermedad"] == option2_slctd]
            dfPieMasculino = dfPieMasculino[dfPieMasculino["Distrito"].isin(lDistritos)]

            dfPieFemenino = dfPieFemenino[dfPieFemenino["Categoria_enfermedad"] == option2_slctd]
            dfPieFemenino = dfPieFemenino[dfPieFemenino["Distrito"].isin(lDistritos)]

            dfPieMasculino=  dfPieMasculino[dfPieMasculino['Sexo']=='Masculino']
            dfPieFemenino=  dfPieFemenino[dfPieFemenino['Sexo']=='Femenino']
            dfPieMasculino = dfPieMasculino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfPieFemenino = dfPieFemenino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfPieMasculino = dfPieMasculino.set_index('Fecha')
            dfPieMasculino = dfPieMasculino.asfreq('MS')
            dfPieMasculino = dfPieMasculino.sort_index()

            dfPieFemenino = dfPieFemenino.set_index('Fecha')
            dfPieFemenino = dfPieFemenino.asfreq('MS')
            dfPieFemenino = dfPieFemenino.sort_index()

            msarimaxMas=modelo_SARIMAX(dfPieMasculino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            msarimaxFem=modelo_SARIMAX(dfPieFemenino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

            pred_dataMas=pd.DataFrame(msarimaxMas[0].predicted_mean)
            pred_dataMas = pred_dataMas.reset_index()
            pred_dataMas.columns=['Fecha','Pronosticos']

            pred_dataFem=pd.DataFrame(msarimaxFem[0].predicted_mean)
            pred_dataFem = pred_dataFem.reset_index()
            pred_dataFem.columns=['Fecha','Pronosticos']

            pred_dataMas.loc[pred_dataMas["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_dataFem.loc[pred_dataFem["Pronosticos"] < 0, "Pronosticos"] = 0

            fig2 = px.pie(values=[pred_dataMas.tail(n=12)['Pronosticos'].sum(),pred_dataFem.tail(n=12)['Pronosticos'].sum()], names=['Masculino','Femenino'], title='Porcentaje de casos pronosticados de '+str(lblEnfermedad).lower()+'<br>por sexo en el año '+str(lblAnio)+' en todos los distritos de Lima',color_discrete_sequence=colors,template='seaborn')

            fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })
            fig2.update_traces(hovertemplate = None,
                  hoverinfo = "skip")
            dicdataCho={}
            ldataCho=[]

            for i in lDistritos:
                dfChoropleth = df.copy()
                ldataCho=[]
                dfChoropleth = dfChoropleth[dfChoropleth["Categoria_enfermedad"] == option2_slctd]
                dfChoropleth=  dfChoropleth[dfChoropleth['Distrito']==i]
                dfChoropleth = dfChoropleth.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
                dfChoropleth = dfChoropleth.set_index('Fecha')
                dfChoropleth = dfChoropleth.asfreq('MS')
                dfChoropleth = dfChoropleth.sort_index()

                msarimaxCho=modelo_SARIMAX(dfChoropleth,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

                pred_dataCho=pd.DataFrame(msarimaxCho[0].predicted_mean)
                pred_dataCho = pred_dataCho.reset_index()
                pred_dataCho.columns=['Fecha','Pronosticos']

                ldataCho.append(pred_dataCho.tail(n=12)['Pronosticos'].sum())
                dicdataCho[i] = ldataCho

            pddatacho = pd.DataFrame.from_dict(dicdataCho)
            pddatachomel = pd.melt(pddatacho, id_vars=None, value_vars=None)

            fig3 = px.choropleth_mapbox(data_frame=pddatachomel,
                geojson=filtered_data,
                locations='variable',
                featureidkey='properties.NOMBDIST',
                color='value',
                color_continuous_scale="oryel",
                mapbox_style="open-street-map",
                center={"lat": -12.04318, "lon": -77.02824},
                zoom=8,
                opacity=0.7,
            )
            fig3.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")
            fig3.update_layout(title={
            'text':'Mapa de incidencias cardiovasculares',
            'font':{'family': 'Arial', 'size': 16},
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            })
            indice_maximo = pddatachomel['value'].idxmax()
            container_max_distrito = pddatachomel.loc[indice_maximo, 'variable']

        elif(option2_slctd!='todo' and option3_slctd!='todo'):
            dfLine = dfLine[dfLine["Categoria_enfermedad"]== option2_slctd]
            dfLine = dfLine[dfLine["Distrito"] == option3_slctd]
            dfLine = dfLine.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLine = dfLine.set_index('Fecha')
            dfLine = dfLine.asfreq('MS')
            dfLine = dfLine.sort_index()

            value=getPDQ(dfLine)[0]
            value_seasonal=getPDQ(dfLine)[1]

            if(option_slctd==2023):
                steps=12
            elif(option_slctd==2024):
                steps=24
            elif(option_slctd==2025):
                steps=36

            msarimax=modelo_SARIMAX(dfLine,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            pred_data=pd.DataFrame(msarimax[0].predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']

            pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))
            fig = px.line(pred_data.tail(n=12), x='Fecha', y='Pronosticos',markers=True,template='seaborn')
            fig.update_layout(title={
            'text': 'Nro. de casos pronosticados con '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en el distrito de "+str(lblDistrito).lower(),
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

            container = int(pred_data.tail(n=12)['Pronosticos'].sum())


            dfPieMasculino = dfPieMasculino[dfPieMasculino["Categoria_enfermedad"]== option2_slctd]
            dfPieMasculino = dfPieMasculino[dfPieMasculino["Distrito"] == option3_slctd]

            dfPieFemenino = dfPieFemenino[dfPieFemenino["Categoria_enfermedad"]== option2_slctd]
            dfPieFemenino = dfPieFemenino[dfPieFemenino["Distrito"] == option3_slctd]

            dfPieMasculino=  dfPieMasculino[dfPieMasculino['Sexo']=='Masculino']
            dfPieFemenino=  dfPieFemenino[dfPieFemenino['Sexo']=='Femenino']
            dfPieMasculino = dfPieMasculino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfPieFemenino = dfPieFemenino.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfPieMasculino = dfPieMasculino.set_index('Fecha')
            dfPieMasculino = dfPieMasculino.asfreq('MS')
            dfPieMasculino = dfPieMasculino.sort_index()

            dfPieFemenino = dfPieFemenino.set_index('Fecha')
            dfPieFemenino = dfPieFemenino.asfreq('MS')
            dfPieFemenino = dfPieFemenino.sort_index()

            msarimaxMas=modelo_SARIMAX(dfPieMasculino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)
            msarimaxFem=modelo_SARIMAX(dfPieFemenino,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

            pred_dataMas=pd.DataFrame(msarimaxMas[0].predicted_mean)
            pred_dataMas = pred_dataMas.reset_index()
            pred_dataMas.columns=['Fecha','Pronosticos']

            pred_dataFem=pd.DataFrame(msarimaxFem[0].predicted_mean)
            pred_dataFem = pred_dataFem.reset_index()
            pred_dataFem.columns=['Fecha','Pronosticos']


            pred_dataMas.loc[pred_dataMas["Pronosticos"] < 0, "Pronosticos"] = 0
            pred_dataFem.loc[pred_dataFem["Pronosticos"] < 0, "Pronosticos"] = 0
            fig2 = px.pie(values=[pred_dataMas.tail(n=12)['Pronosticos'].sum(),pred_dataFem.tail(n=12)['Pronosticos'].sum()], names=['Masculino','Femenino'], title='Porcentaje de casos pronosticados de '+str(lblEnfermedad).lower()+'<br>por sexo en el año '+str(lblAnio)+' en '+str(lblDistrito).lower(),color_discrete_sequence=colors,template='seaborn')

            fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })
            fig2.update_traces(hovertemplate = None,
                  hoverinfo = "skip")
            dicdataCho={}
            ldataCho=[]
            lDistritos=[option3_slctd]
            for i in lDistritos:
                dfChoropleth = df.copy()
                ldataCho=[]
                dfChoropleth = dfChoropleth[dfChoropleth["Categoria_enfermedad"]== option2_slctd]
                dfChoropleth=  dfChoropleth[dfChoropleth['Distrito']==i]
                dfChoropleth = dfChoropleth.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
                dfChoropleth = dfChoropleth.set_index('Fecha')
                dfChoropleth = dfChoropleth.asfreq('MS')
                dfChoropleth = dfChoropleth.sort_index()

                msarimaxCho=modelo_SARIMAX(dfChoropleth,str(option_slctd)+'-01-01',steps,True,value,value_seasonal)

                pred_dataCho=pd.DataFrame(msarimaxCho[0].predicted_mean)
                pred_dataCho = pred_dataCho.reset_index()
                pred_dataCho.columns=['Fecha','Pronosticos']

                ldataCho.append(pred_dataCho.tail(n=12)['Pronosticos'].sum())
                dicdataCho[i] = ldataCho

            pddatacho = pd.DataFrame.from_dict(dicdataCho)
            pddatachomel = pd.melt(pddatacho, id_vars=None, value_vars=None)

            fig3 = px.choropleth_mapbox(data_frame=pddatachomel,
                geojson=filtered_data,
                locations='variable',
                featureidkey='properties.NOMBDIST',
                color='value',
                color_continuous_scale="oryel",
                mapbox_style="open-street-map",
                center={"lat": -12.04318, "lon": -77.02824},
                zoom=8,
                opacity=0.7,
            )
            fig3.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")
            fig3.update_layout(title={
            'text':'Mapa de incidencias cardiovasculares',
            'font':{'family': 'Arial', 'size': 16},
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            })
            indice_maximo = pddatachomel['value'].idxmax()
            container_max_distrito = pddatachomel.loc[indice_maximo, 'variable']

    elif(option_slctd<2023):

        entrenarPas=False
        filtro = (dfModelos['anio'] == int(option_slctd)) & (dfModelos['enfermedad'] == option2_slctd) & (dfModelos['distrito'] == option3_slctd)
        dfModelofiltrado = dfModelos.loc[filtro, ['value', 'valueSeasonal']]

        if dfModelofiltrado.empty:
            entrenarPas=True
        else:
            entrenarPas=False
            indice = dfModelofiltrado.index.tolist()
            indice = int(indice[0])

            value = dfModelofiltrado.loc[indice,'value']
            value_seasonal = dfModelofiltrado.loc[indice,'valueSeasonal']

            value=tuple(map(int,f'{value:03d}'))
            value_seasonal=str(value_seasonal)

            lst = [int(c) for c in value_seasonal]
            t1 = tuple(lst[:3])
            t2 = convertString(lst[3:])
            t2 = list((t2,))
            t2 = [int(elem) for elem in t2]
            t2 = tuple(t2)
            value_seasonal = t1 + t2

        if(option2_slctd=='todo' and option3_slctd!='todo'):
            dff = dff[dff["Anio"] == option_slctd]
            dff = dff[dff["Categoria_enfermedad"].isin(lEnfermedades)]
            dff = dff[dff["Distrito"] == option3_slctd]
            dff = dff.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfLineAll = dfLineAll[dfLineAll["Anio"].isin([option_slctd-1,option_slctd])]
            dfLineAll = dfLineAll[dfLineAll["Categoria_enfermedad"].isin(lEnfermedades)]
            dfLineAll = dfLineAll[dfLineAll["Distrito"] == option3_slctd]
            dfLineAll = dfLineAll.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLineAll = dfLineAll.set_index('Fecha')
            dfLineAll = dfLineAll.asfreq('MS')
            dfLineAll = dfLineAll.sort_index()

            if(entrenarPas==True):
                value=getPDQ(dfLineAll)[0]
                value_seasonal=getPDQ(dfLineAll)[1]


            result=modelo_SARIMAX_past(dfLineAll,value,value_seasonal)
            predictionPast = result.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)

            pred_data=pd.DataFrame(predictionPast.predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']


            dff2 = dff2[dff2["Anio"] == option_slctd]
            dff2 = dff2[dff2["Categoria_enfermedad"].isin(lEnfermedades)]
            dff2 = dff2[dff2["Distrito"] == option3_slctd]
            dff2 = dff2.groupby(['Sexo','Categoria_enfermedad'])['Nro_casos'].sum().reset_index()

            dff3 = dff3[dff3["Anio"] == option_slctd]
            dff3 = dff3[dff3["Categoria_enfermedad"].isin(lEnfermedades)]
            dff3 = dff3[dff3["Distrito"] == option3_slctd]
            dff3 = dff3.groupby(['Anio','Categoria_enfermedad'])['Nro_casos'].sum().reset_index()

            dff4 = dff4[dff4["Anio"] == option_slctd]
            dff4 = dff4[dff4["Categoria_enfermedad"].isin(lEnfermedades)]
            dff4 = dff4[dff4["Distrito"] == option3_slctd]
            dff4 = dff4.groupby(['Anio','Distrito'])['Nro_casos'].sum().reset_index()

        elif(option3_slctd=='todo' and option2_slctd!='todo'):
            dff = dff[dff["Anio"] == option_slctd]
            dff = dff[dff["Categoria_enfermedad"] == option2_slctd]
            dff = dff[dff["Distrito"].isin(lDistritos)]
            dff = dff.groupby(['Fecha'])['Nro_casos'].sum().reset_index()


            dfLineAll = dfLineAll[dfLineAll["Anio"].isin([option_slctd-1,option_slctd])]
            dfLineAll = dfLineAll[dfLineAll["Categoria_enfermedad"] == option2_slctd]
            dfLineAll = dfLineAll[dfLineAll["Distrito"].isin(lDistritos)]
            dfLineAll = dfLineAll.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLineAll = dfLineAll.set_index('Fecha')
            dfLineAll = dfLineAll.asfreq('MS')
            dfLineAll = dfLineAll.sort_index()

            result=modelo_SARIMAX_past(dfLineAll,value,value_seasonal)
            predictionPast = result.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)

            pred_data=pd.DataFrame(predictionPast.predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']


            dff2 = dff2[dff2["Anio"] == option_slctd]
            dff2 = dff2[dff2["Categoria_enfermedad"] == option2_slctd]
            dff2 = dff2[dff2["Distrito"].isin(lDistritos)]
            dff2 = dff2.groupby(['Sexo'])['Nro_casos'].sum().reset_index()

            dff3 = dff3[dff3["Anio"] == option_slctd]
            dff3 = dff3[dff3["Categoria_enfermedad"] == option2_slctd]
            dff3 = dff3[dff3["Distrito"].isin(lDistritos)]
            dff3 = dff3.groupby(['Anio'])['Nro_casos'].sum().reset_index()

            dff4 = dff4[dff4["Anio"] == option_slctd]
            dff4 = dff4[dff4["Categoria_enfermedad"] == option2_slctd]
            dff4 = dff4[dff4["Distrito"].isin(lDistritos)]
            dff4 = dff4.groupby(['Anio','Provincia','Distrito'])['Nro_casos'].sum().reset_index()
        elif(option2_slctd=='todo' and option3_slctd=='todo'):
            dff = dff[dff["Anio"] == option_slctd]
            dff = dff[dff["Categoria_enfermedad"].isin(lEnfermedades)]
            dff = dff[dff["Distrito"].isin(lDistritos)]
            dff = dff.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfLineAll = dfLineAll[dfLineAll["Anio"].isin([option_slctd-1,option_slctd])]
            dfLineAll = dfLineAll[dfLineAll["Categoria_enfermedad"].isin(lEnfermedades)]
            dfLineAll = dfLineAll[dfLineAll["Distrito"].isin(lDistritos)]
            dfLineAll = dfLineAll.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLineAll = dfLineAll.set_index('Fecha')
            dfLineAll = dfLineAll.asfreq('MS')
            dfLineAll = dfLineAll.sort_index()

            result=modelo_SARIMAX_past(dfLineAll,value,value_seasonal)
            predictionPast = result.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)

            pred_data=pd.DataFrame(predictionPast.predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']


            dff2 = dff2[dff2["Anio"] == option_slctd]
            dff2 = dff2[dff2["Categoria_enfermedad"].isin(lEnfermedades)]
            dff2 = dff2[dff2["Distrito"].isin(lDistritos)]
            dff2 = dff2.groupby(['Sexo'])['Nro_casos'].sum().reset_index()

            dff3 = dff3[dff3["Anio"] == option_slctd]
            dff3 = dff3[dff3["Categoria_enfermedad"].isin(lEnfermedades)]
            dff3 = dff3[dff3["Distrito"].isin(lDistritos)]
            dff3 = dff3.groupby(['Anio'])['Nro_casos'].sum().reset_index()

            dff4 = dff4[dff4["Anio"] == option_slctd]
            dff4 = dff4[dff4["Categoria_enfermedad"].isin(lEnfermedades)]
            dff4 = dff4[dff4["Distrito"].isin(lDistritos)]
            dff4 = dff4.groupby(['Anio','Provincia','Distrito'])['Nro_casos'].sum().reset_index()
        else:
            dff = dff[dff["Anio"] == option_slctd]
            dff = dff[dff["Categoria_enfermedad"] == option2_slctd]
            dff = dff[dff["Distrito"] == option3_slctd]
            dff = dff.groupby(['Fecha'])['Nro_casos'].sum().reset_index()

            dfLineAll = dfLineAll[dfLineAll["Anio"].isin([option_slctd-1,option_slctd])]
            dfLineAll = dfLineAll[dfLineAll["Categoria_enfermedad"] == option2_slctd]
            dfLineAll = dfLineAll[dfLineAll["Distrito"] == option3_slctd]
            dfLineAll = dfLineAll.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dfLineAll = dfLineAll.set_index('Fecha')
            dfLineAll = dfLineAll.asfreq('MS')
            dfLineAll = dfLineAll.sort_index()

            value=getPDQ(dfLineAll)[0]
            value_seasonal=getPDQ(dfLineAll)[1]

            result=modelo_SARIMAX_past(dfLineAll,value,value_seasonal)
            predictionPast = result.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)
            pred_data=pd.DataFrame(predictionPast.predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']
            dff2 = dff2[dff2["Anio"] == option_slctd]
            dff2 = dff2[dff2["Categoria_enfermedad"] == option2_slctd]
            dff2 = dff2[dff2["Distrito"] == option3_slctd]
            dff2 = dff2.groupby(['Sexo'])['Nro_casos'].sum().reset_index()

            dff3 = dff3[dff3["Anio"] == option_slctd]
            dff3 = dff3[dff3["Categoria_enfermedad"] == option2_slctd]
            dff3 = dff3[dff3["Distrito"] == option3_slctd]

            dff4 = dff4[dff4["Anio"] == option_slctd]
            dff4 = dff4[dff4["Categoria_enfermedad"] == option2_slctd]
            dff4 = dff4[dff4["Distrito"] == option3_slctd]

            dff4 = dff4.groupby(['Anio','Provincia','Distrito'])['Nro_casos'].sum().reset_index()

        container = "Número de enfermos total: {}".format(dff3['Nro_casos'].sum())
        pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
        pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))
        fig = px.line(template='seaborn',line_shape='spline')
        fig.add_scatter(x=dff['Fecha'], y=dff['Nro_casos'],mode='lines+markers', name='Datos reales',line=dict(color='#084B8A'))
        fig.add_scatter(x=pred_data['Fecha'], y=pred_data['Pronosticos'], mode='lines+markers', name='Dato pronosticado',line=dict(color='#EE4E5A'))
        fig.update_layout(yaxis=dict(range=[0, None]))

        if(option2_slctd=='todo' and option3_slctd=='todo'):
            fig.update_layout(title={
            'text': 'Total de casos de enfermedades cardiovasculares reales VS pronosticados<br>'+"en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))
        elif(option2_slctd=='todo' and option3_slctd!='todo'):
            fig.update_layout(title={
            'text': "Total de casos de enfermedades cardiovasculares reales VS pronosticados<br>en el año "+str(lblAnio)+" en el distrito de "+str(lblDistrito).lower(),
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))
        elif(option2_slctd!='todo' and option3_slctd=='todo'):
            fig.update_layout(title={
            'text': 'Nro. de casos reales VS pronosticados con '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))
        elif(option2_slctd!='todo' and option3_slctd!='todo'):
            fig.update_layout(title={
            'text': 'Nro. de casos reales VS pronosticados con '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en el distrito de "+str(lblDistrito).lower(),
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

        dff2.loc[dff2["Nro_casos"] < 0, "Nro_casos"] = 0
        if(option2_slctd=='todo' and option3_slctd=='todo'):
            fig2 = px.pie(dff2, values='Nro_casos', names='Sexo', title='Porcentaje de casos reales de enfermedades cardiovasculares<br>por sexo en el año '+str(lblAnio)+' en todos los distritos de Lima',color_discrete_sequence=colors,template='seaborn')

        elif(option2_slctd=='todo' and option3_slctd!='todo'):
            fig2 = px.pie(dff2, values='Nro_casos', names='Sexo', title='Porcentaje de casos reales de enfermedades cardiovasculares<br>por sexo en el año '+str(lblAnio)+' en '+str(lblDistrito).lower(),color_discrete_sequence=colors,template='seaborn')

        elif(option2_slctd!='todo' and option3_slctd=='todo'):
            fig2 = px.pie(dff2, values='Nro_casos', names='Sexo', title='Porcentaje de casos reales de '+str(lblEnfermedad).lower()+'<br>por sexo en el año '+str(lblAnio)+' en todos los distritos de Lima',color_discrete_sequence=colors,template='seaborn')

        elif(option2_slctd!='todo' and option3_slctd!='todo'):
            fig2 = px.pie(dff2, values='Nro_casos', names='Sexo', title='Porcentaje de casos reales de '+str(lblEnfermedad).lower()+'<br>por sexo en el año '+str(lblAnio)+' en '+str(lblDistrito).lower(),color_discrete_sequence=colors,template='seaborn')


        fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })


        fig3 = px.choropleth_mapbox(data_frame=dff4,
                        geojson=filtered_data,
                        locations='Distrito',
                        featureidkey='properties.NOMBDIST',
                        color='Nro_casos',
                        color_continuous_scale="oryel",
                        mapbox_style="open-street-map",
                        center={"lat": -12.04318, "lon": -77.02824},
                        zoom=8,
                        opacity=0.7,
                    )
        fig3.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
        fig3.update_layout(title={
            'text':'Mapa de incidencias cardiovasculares',
            'font':{'family': 'Arial', 'size': 16},
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
        indice_maximo = dff4['Nro_casos'].idxmax()
        container_max_distrito = dff4.loc[indice_maximo, 'Distrito']

    header = navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Dropdown(id="slct_year",
                            options=[
                                {"label": "2014", "value": 2014},
                                {"label": "2015", "value": 2015},
                                {"label": "2016", "value": 2016},
                                {"label": "2017", "value": 2017},
                                {"label": "2018", "value": 2018},
                                {"label": "2019", "value": 2019},
                                {"label": "2020", "value": 2020},
                                {"label": "2021", "value": 2021},
                                {"label": "2022", "value": 2022},
                                {"label": "2023", "value": 2023},
                                {"label": "2024", "value": 2024},
                                {"label": "2025", "value": 2025}],
                                multi=False,
                                value=option_slctd,
                                style = dict(
                                width = '120px',
                                margin = '0px 10px 0px 0px',
                                )
            )
        ),
        dbc.NavItem(
            dcc.Dropdown(id="slct_disease",
                    options=[
                        {"label": "Seleccionar todo", "value": 'todo'},
                        {"label": "Hipertensión esencial (Primaria)", "value": 'HIPERTENSION ESENCIAL (PRIMARIA)'},
                        {"label": "Hipertensión secundaria", "value": 'HIPERTENSION SECUNDARIA'},
                        {"label": "Otras arritmias cardiacas", "value": 'OTRAS ARRITMIAS CARDIACAS'},
                        {"label": "Venas varicosas de los miembros inferiores", "value": 'VENAS VARICOSAS DE LOS MIEMBROS INFERIORES'}],
                    multi=False,
                    value=option2_slctd,
                    style = dict(
                                width = '320px',
                                margin = '0px 10px 0px 0px',
                            )
            )
        ),
        dbc.NavItem(
            dcc.Dropdown(id="slct_distrito",
                    options=[
                        {"label": "Seleccionar todo", "value": 'todo'},
                        {"label": "Ancón", "value": 'ANCON'},
                        {"label": "ate", "value": 'ATE'},
                        {"label": "Barranco", "value": 'BARRANCO'},
                        {"label": "Breña", "value": 'BREÑA'},
                        {"label": "Carabayllo", "value": 'CARABAYLLO'},
                        {"label": "Chaclacayo", "value": 'CHACLACAYO'},
                        {"label": "Chorrillos", "value": 'CHORRILLOS'},
                        {"label": "Cieneguilla", "value": 'CIENEGUILLA'},
                        {"label": "Comas", "value": 'COMAS'},
                        {"label": "El agustino", "value": 'EL AGUSTINO'},
                        {"label": "Independencia", "value": 'INDEPENDENCIA'},
                        {"label": "Jesus maria", "value": 'JESUS MARIA'},
                        {"label": "La molina", "value": 'LA MOLINA'},
                        {"label": "La victoria", "value": 'LA VICTORIA'},
                        {"label": "Lima", "value": 'LIMA'},
                        {"label": "Lince", "value": 'LINCE'},
                        {"label": "Los olivos", "value": 'LOS OLIVOS'},
                        {"label": "Lurigancho", "value": 'LURIGANCHO'},
                        {"label": "Lurin", "value": 'LURIN'},
                        {"label": "Magdalena del mar", "value": 'MAGDALENA DEL MAR'},
                        {"label": "Miraflores", "value": 'MIRAFLORES'},
                        {"label": "Pachacamac", "value": 'PACHACAMAC'},
                        {"label": "Pucusana", "value": 'PUCUSANA'},
                        {"label": "Pueblo libre", "value": 'PUEBLO LIBRE'},
                        {"label": "Puente piedra", "value": 'PUENTE PIEDRA'},
                        {"label": "Punta hermosa", "value": 'PUNTA HERMOSA'},
                        {"label": "Punta negra", "value": 'PUNTA NEGRA'},
                        {"label": "Rimac", "value": 'RIMAC'},
                        {"label": "San bartolo", "value": 'SAN BARTOLO'},
                        {"label": "San borja", "value": 'SAN BORJA'},
                        {"label": "San isidro", "value": 'SAN ISIDRO'},
                        {"label": "San juan de lurigancho", "value": 'SAN JUAN DE LURIGANCHO'},
                        {"label": "San juan de miraflores", "value": 'SAN JUAN DE MIRAFLORES'},
                        {"label": "San luis", "value": 'SAN LUIS'},
                        {"label": "San martin de porres", "value": 'SAN MARTIN DE PORRES'},
                        {"label": "San miguel", "value": 'SAN MIGUEL'},
                        {"label": "Santa anita", "value": 'SANTA ANITA'},
                        {"label": "Santa maria del mar", "value": 'SANTA MARIA DEL MAR'},
                        {"label": "Santa rosa", "value": 'SANTA ROSA'},
                        {"label": "Santiago de surco", "value": 'SANTIAGO DE SURCO'},
                        {"label": "Surquillo", "value": 'SURQUILLO'},
                        {"label": "Villa el salvador", "value": 'VILLA EL SALVADOR'},
                        {"label": "Villa maria del triunfo", "value": 'VILLA MARIA DEL TRIUNFO'},
                        {"label": "Villa maria del triunfo", "value": 'VILLA MARIA DEL TRIUNFO'}],
                    multi=False,
                    value=option3_slctd,
                    style = dict(
                                width = '120px',
                                )
            )
        ),
        dbc.NavItem(
            dbc.Button("Analizar", color="light", className="me-1", id='submit-button-state', n_clicks=0)
        )
    ],
    brand="CardioWatch",
    brand_href="#",
    color="dark",
    dark=True,
    )

    card_content = [
    dbc.CardBody(
        [
            html.H5("Total de enfermos", className="card-title"),
            html.P(
                className="card-text",id="output_container",children=container
            ),
        ]
    ),
    ]

    card_content2 = [
        dbc.CardBody(
            [
                html.H5("Distrito mayor enfermos", className="card-title"),
                html.P(
                    className="card-text",id="output_container_max_distrito",children=container_max_distrito
                ),
            ]
        ),
    ]

    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content, color="dark", inverse=True)),
                    dbc.Col(dbc.Card(card_content2, color="dark", inverse=True), className="card2"),
                ],className='cardline'
            )
        ],style={"width": "100%","margin":"0px","display":"flex"},className="container carddata"
    )


    layout=header,dbc.Container(className="p-3"),cards,dbc.Row([dbc.Col(html.Div([html.Div([dcc.Graph(id='lineal_graph',figure=fig,className="figure_lineal")], className="graph_lineal")]), lg=7,sm=12),dbc.Col(html.Div([html.Div([dcc.Graph(id='circular_graph',figure=fig2)])]), lg=4,sm=12)],className="container-fluid mt-5"),html.Div([dbc.Row([dbc.Col(html.Div([dcc.Graph(id='map_graph',figure=fig3)]), sm=12)],className="container-fluid")], className="my_map")
    return (container,container_max_distrito,fig,fig2,fig3,layout)

######### Función para actualizar dropdown de enfermedades #########
def update_dropdown_2(value,previous_value):
        if value != 'todo':
            return 'todo'
        else:
            return previous_value

######### Función para actualizar dropdown de distritos #########
def update_dropdown_3(value,previous_value):
        if value != 'todo':
            return 'todo'
        else:
            return previous_value

######### Función para navbar responsive #########
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

def init_callbacks(dash_app):
    dash_app.callback(
    [Output(component_id='output_container', component_property='children'),
    Output(component_id='output_container_max_distrito', component_property='children'),
    Output(component_id='lineal_graph', component_property='figure'),
    Output(component_id='circular_graph', component_property='figure'),
    Output(component_id='map_graph', component_property='figure'),
    Output(component_id="loading-output-1",component_property="children")],
    [Input('submit-button-state', 'n_clicks')],
    [State(component_id='slct_year', component_property='value'),
    State(component_id='slct_disease', component_property='value'),
    State(component_id='slct_distrito', component_property='value')]
    )(update_graph)

    dash_app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
    )(toggle_navbar_collapse)

    dash_app.callback(
    Output("slct_disease", "value"),
    [Input("slct_distrito", "value")],
    [State('slct_disease', 'value')],
    )(update_dropdown_2)

    dash_app.callback(
    Output("slct_distrito", "value"),
    [Input("slct_disease", "value")],
    [State('slct_distrito', 'value')],
    )(update_dropdown_3)

def create_dash_application(flask_app):
    dash_app = dash.Dash(server=flask_app, name='Dashboard', url_base_pathname='/dash/',external_stylesheets=[dbc.themes.BOOTSTRAP,'https://static.staticsave.com/assets/style-css.css'])

    dash_app.layout = html.Div([

    dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div([
                header,
                dbc.Container(className="p-3"),
                cards,
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dcc.Graph(id='lineal_graph',figure={},className="figure_lineal")
                                ], className="graph_lineal")
                        ]), lg=7,sm=12),
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dcc.Graph(id='circular_graph')
                            ])
                        ]), lg=4,sm=12)
                ],className="container-fluid mt-5"),
                html.Div([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                dcc.Graph(id='map_graph')
                            ]), sm=12
                        )
                    ],className="container-fluid")
                ], className="my_map")
            ],id="loading-output-1",style={"height": "100vh"})
    )], className='dash_all')
    init_callbacks(dash_app)
    return dash_app.server