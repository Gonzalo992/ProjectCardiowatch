import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash
import statsmodels.api as sm
import itertools
import math
import dash_bootstrap_components as dbc

app = Dash(__name__,suppress_callback_exceptions=True)


######### Carga de dataset #########
df = pd.read_csv("https://raw.githubusercontent.com/Gonzalo992/dataset/main/Datos_definitivo.csv",sep=',',encoding='latin-1')


######### Preparación de Dataset #########
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')

######### Components HTML #########
header = navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Dropdown(id="slct_year",
                            options=[
                                {"label": "2019", "value": 2019}],
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
                        {"label": "Hipertensión esencial (Primaria)", "value": 'HIPERTENSION ESENCIAL (PRIMARIA)'}],
                    multi=False,
                    value='HIPERTENSION ESENCIAL (PRIMARIA)',
                    style = dict(
                                width = '320px',
                                margin = '0px 10px 0px 0px',
                            )
            )
        ),
        dbc.NavItem(
            dcc.Dropdown(id="slct_distrito",
                    options=[
                        {"label": "Seleccionar todo", "value": 'todo'}],
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


######### Get PDQ Estático ########
dfOtro = df.copy()
dfOtro = dfOtro.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
dfOtro = dfOtro.set_index('Fecha')
dfOtro = dfOtro.asfreq('MS')
dfOtro = dfOtro.sort_index()

######### Función modelo SARIMAX para datos pasados #########
def modelo_SARIMAX_past(datos,value,value_seasonal):
    model = sm.tsa.statespace.SARIMAX(datos, order = value,seasonal_order = value_seasonal)
    result = model.fit()
    return result

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
    dfLineAll = df.copy()
    dffLineOther= df.copy()



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
    dfModelos = dfModelos.dropna()
    if(option_slctd<2023):
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


        if(option3_slctd=='todo' and option2_slctd!='todo'):
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

            dffLineOther = dffLineOther[dffLineOther["Anio"].isin([option_slctd-1,option_slctd])]
            dffLineOther = dffLineOther.groupby(['Fecha'])['Nro_casos'].sum().reset_index()
            dffLineOther = dffLineOther.set_index('Fecha')
            dffLineOther = dffLineOther.asfreq('MS')
            dffLineOther = dffLineOther.sort_index()


            #value=getPDQ(dfLineAll)[0]
            #value_seasonal=getPDQ(dfLineAll)[1]
            valueOther=getPDQ(dffLineOther)[0]
            value_seasonalOther=getPDQ(dffLineOther)[1]

            result=modelo_SARIMAX_past(dfLineAll,value,value_seasonal)

            resultOther=modelo_SARIMAX_past(dffLineOther,valueOther,value_seasonalOther)

            predictionPast = result.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)

            predictionPastOther = resultOther.get_prediction(start = pd.to_datetime(str(option_slctd)+'-01-01'), dynamic = False)

            pred_data=pd.DataFrame(predictionPast.predicted_mean)
            pred_data = pred_data.reset_index()
            pred_data.columns=['Fecha','Pronosticos']


            pred_dataOther=pd.DataFrame(predictionPastOther.predicted_mean)
            pred_dataOther = pred_dataOther.reset_index()
            pred_dataOther.columns=['Fecha','Pronosticos']

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

        container = "Número de enfermos total: {}".format(dff3['Nro_casos'].sum())
        pred_data.loc[pred_data["Pronosticos"] < 0, "Pronosticos"] = 0
        pred_data["Pronosticos"] = pred_data["Pronosticos"].apply(lambda x: math.trunc(x))


        pred_dataOther.loc[pred_dataOther["Pronosticos"] < 0, "Pronosticos"] = 0
        pred_dataOther["Pronosticos"] = pred_dataOther["Pronosticos"].apply(lambda x: math.trunc(x))


        fig = px.line(template='seaborn',line_shape='spline')
        fig.add_scatter(x=dff['Fecha'], y=dff['Nro_casos'],mode='lines+markers', name='Datos reales',line=dict(color='#084B8A'))
        fig.add_scatter(x=pred_data['Fecha'], y=pred_data['Pronosticos'], mode='lines+markers', name='Cardiowatch pronosticado',line=dict(color='#EE4E5A'))
        fig.add_scatter(x=pred_dataOther['Fecha'], y=pred_dataOther['Pronosticos'], mode='lines+markers', name='SARIMAX no segmentado',line=dict(color='#008f39'))
        fig.update_layout(yaxis=dict(range=[0, None]))

        if(option2_slctd!='todo' and option3_slctd=='todo'):
            fig.update_layout(title={
            'text': 'Nro. de casos reales VS pronosticados con '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'font': {'family': 'Arial', 'size': 13},
            },
            xaxis_title='Fecha',
            yaxis_title='Número de casos',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1))

        dff2.loc[dff2["Nro_casos"] < 0, "Nro_casos"] = 0

        if(option2_slctd!='todo' and option3_slctd=='todo'):
            #fig2 = px.pie(dff2, values='Nro_casos', names='Sexo', title='Porcentaje de casos reales de '+str(lblEnfermedad).lower()+'<br>por sexo en el año '+str(lblAnio)+' en todos los distritos de Lima',color_discrete_sequence=colors,template='seaborn')
            dff_dispersion=dff.copy()
            pred_data_dispersion=pred_data.copy()
            pred_dataOther_dispersion=pred_dataOther.copy()

            pred_data_dispersion.rename(columns={'Pronosticos': 'Nro_casos'}, inplace=True)
            pred_dataOther_dispersion.rename(columns={'Pronosticos': 'Nro_casos'}, inplace=True)

            dff_dispersion['tipo']='Datos reales'
            pred_data_dispersion['tipo']='Cardiowatch'
            pred_dataOther_dispersion['tipo']='Otro modelo'
            df_dispersion = pd.concat([dff_dispersion,pred_data_dispersion, pred_dataOther_dispersion])
            fig2 = px.scatter(df_dispersion, x="Fecha", y="Nro_casos",color="tipo")
            fig2.update_layout(title={
            'text': 'Gráfico de dispersión de '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'x':0.5,
            'y':0.95,
            'font': {'family': 'Arial', 'size': 13},
            },legend=dict(orientation='h', yanchor='bottom', xanchor='left', x=0, y=-0.2,title=None))


            df_error= pd.concat([dff_dispersion,pred_data_dispersion])
            df_error_real= df_error[df_error['tipo']== 'Datos reales']
            df_error_pronosticado= df_error[df_error['tipo']== 'Cardiowatch']

            df_error['error']=df_error_real['Nro_casos']-df_error_pronosticado['Nro_casos']
            df_error['clase'] = 'Error cardiowatch'
            df_error = df_error.loc[df_error['tipo'] == 'Datos reales']




            df_errorOther= pd.concat([dff_dispersion,pred_dataOther_dispersion])
            df_error_realOther= df_errorOther[df_errorOther['tipo']== 'Datos reales']
            df_error_pronosticadoOther= df_errorOther[df_errorOther['tipo']== 'Otro modelo']

            df_errorOther['error']=df_error_realOther['Nro_casos']-df_error_pronosticadoOther['Nro_casos']
            df_errorOther['clase'] = 'Error otro modelo'
            df_errorOther = df_errorOther.loc[df_errorOther['tipo'] == 'Datos reales']

            df_errorTotal = pd.concat([df_error, df_errorOther], ignore_index=True)

            fig3 = px.scatter(df_errorTotal, x="Nro_casos", y="error",color="clase")
            fig3.update_layout(title={
            'text': 'Gráfico de errores de predicción de '+str(lblEnfermedad).lower()+"<br>en el año "+str(lblAnio)+" en todos los distritos de Lima",
            'x':0.5,
            'y':0.95,
            'font': {'family': 'Arial', 'size': 13},
            },legend=dict(orientation='h', yanchor='bottom', xanchor='left', x=0, y=-0.2,title=None))
        fig2.update_layout(title={
            'font': {'family': 'Arial', 'size': 13},
            })

        indice_maximo = dff4['Nro_casos'].idxmax()
        container_max_distrito = dff4.loc[indice_maximo, 'Distrito']

    header = navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Dropdown(id="slct_year",
                            options=[
                                {"label": "2019", "value": 2019}],
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
                        {"label": "Hipertensión esencial (Primaria)", "value": 'HIPERTENSION ESENCIAL (PRIMARIA)'}],
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
                        {"label": "Seleccionar todo", "value": 'todo'}],
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


    layout=header,dbc.Container(className="p-3"),dbc.Row([dbc.Col(html.Div([html.Div([dcc.Graph(id='lineal_graph',figure=fig,className="figure_lineal")], className="graph_lineal")]), lg=6,sm=12),dbc.Col(html.Div([html.Div([dcc.Graph(id='dispersion_graph',figure=fig2)])]), lg=5,sm=12)],className="container-fluid mt-5"),dbc.Row([dbc.Col(html.Div([html.Div([dcc.Graph(id='error_graph',figure=fig3)])]), lg=12,sm=12)],className="container-fluid mt-5"),


    return (fig,fig2,fig3,layout)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

def init_callbacks(dash_app):

    dash_app.callback(
    [
    Output(component_id='lineal_graph', component_property='figure'),
    Output(component_id='dispersion_graph', component_property='figure'),
    Output(component_id='error_graph', component_property='figure'),
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

def create_validacion_application(flask_app):
    dash_app = dash.Dash(server=flask_app, name="Validacion", url_base_pathname="/validacion/",external_stylesheets=[dbc.themes.BOOTSTRAP,'https://static.staticsave.com/assets/style-css.css'])
    dash_app.layout = html.Div([
    dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div([
                header,
                dbc.Container(className="p-3"),
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dcc.Graph(id='lineal_graph',figure={},className="figure_lineal")
                                ], className="graph_lineal")
                        ]), lg=6,sm=12),
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dcc.Graph(id='dispersion_graph')
                            ])
                        ]), lg=5,sm=12)
                ],className="container-fluid mt-5"),
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dcc.Graph(id='error_graph')
                            ])
                        ]), lg=12,sm=12)
                ],className="container-fluid mt-5"),
            ],id="loading-output-1")
    )], className='dash_all')
    init_callbacks(dash_app)
    return dash_app.server