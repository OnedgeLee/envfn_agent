import argparse
import os
import sys
import subprocess
import signal
import json
import numpy as np
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from source_data import Mongo
from utils import DashBuffer
from pymongo import MongoClient

port = 9000

# ctrl = subprocess.Popen('python data_control.py', shell=True)

# def signal_handler(signal, frame):
#     ctrl.terminate()
#     print('===== Controller Terminated =====')
#     print('===== Plotter closing =====')
#     sys.exit()

# signal.signal(signal.SIGINT, signal_handler)

source = Mongo(MongoClient("mongodb://localhost:27017"))
DB = 'data_center'
COLLECTION = 'obs'
DICT = {
    'agent_east_temp': 'AI적용 동쪽서버실온도(C)',
    'agent_west_temp': 'AI적용 서쪽서버실온도(C)',
    'agent_hvac_power': 'AI적용 공조전력(W)',
    'base_east_temp': 'AI미적용 동쪽서버실온도(C)',
    'base_west_temp': 'AI미적용 서쪽서버실온도(C)',
    'base_hvac_power': 'AI미적용 공조전력(W)',
    'hvac_difference': '공조전력차(W)',
    'agent_bill': 'AI적용 전기요금(억 원)',
    'base_bill': 'AI미적용 전기요금(억 원)',
    'bill_difference': '전기요금차(억 원)',
    'agent_hvac_energy': 'AI적용 공조전력량(kWh)',
    'base_hvac_energy': 'AI미적용 공조전력량(kWh)',
    'hvac_energy_difference': '공조전력량차(kWh)',
    'agent_hvac_energy_acc': 'AI적용 누적 공조전력량(kWh)',
    'base_hvac_energy_acc': 'AI미적용 누적 공조전력량(kWh)',
    'hvac_energy_difference_acc': '누적 공조전력량차(kWh)'
}

app = dash.Dash('Streaming-Observation')
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.Div([
        html.Img(src='/assets/aidentify_logo.png',
                 style={
                    'width':'auto',
                    'height': '55px',
                    'float': 'right',
                    'position': 'relative',
                    'margin-top': '5px',
                    'margin-bottom': '5px',
                },
                ),
        html.H2('데이터센터 AI 제어',
                style={
                    'position': 'relative',
                    'top': '15px',
                    'display': 'inline',
                    'font-size': '3rem',
                    'font-weight':'normal'
                }),
    ], className='row twelve columns banner'),

    html.Div([
        html.Div([
            dcc.RadioItems(
                id='select-data',
                options=[
                    {'label': 'HVAC Energy', 'value':'hvac_energy'},
                    {'label': 'HVAC Energy Accum', 'value':'hvac_energy_acc'},
                    {'label': 'Electricity Bill', 'value':'elec_bill'},
                    {'label': 'East Temperature', 'value':'east_temp'},
                    {'label': 'West Temperature', 'value':'west_temp'},
                    {'label': 'HVAC Power', 'value':'hvac_power'},                    
                ],
                value='HVAC Power',
                labelStyle={'display':'inline-block'}
            ),
        ], className='twelve columns')
    ], className='row' ,style={ 
        'margin-left':'40px', 
        'margin-right':'40px', 
        'margin-top':'20px', 
       }
    ),
                                                              
    html.Div([
        html.Div(id='bill', className='bill-info'),
        html.Div([
            dcc.Graph(id='observation-graph'),
        ], className='twelve columns', style=dict(textAlign='center')),
        dcc.Interval(id='observation-update', interval=1500, n_intervals=0),
    ], className='row observation-row', style={ 'margin-left':'40px',
                                             'margin-right':'40px',
                                             'margin-top':'20px',
                                             'margin-bottom':'20px'}),


], className='container')

@app.callback(
    Output(component_id='observation-graph', component_property='figure'), 
    [Input(component_id='observation-update', component_property='n_intervals'),
     Input(component_id='select-data', component_property='value')]
)
def update_graph(n_intervals, data_name_set):

    data = source.fetch(DB, COLLECTION).toDf()
    if not data.empty:
        hour = data['hour'].values
        month = np.ceil(hour[data['hour'].index % 730 == 0] / 730).astype(int)
        month = [str(i) + 'M' for i in month]
        agent = list()
        data_names = list()
        measure = None
        if data_name_set == 'hvac_power':
            data_names = ['base_hvac_power','agent_hvac_power','hvac_difference']
            minY = -500
            maxY = 3000
            measure = '(W)'
        elif data_name_set == 'east_temp':
            data_names = ['base_east_temp','agent_east_temp']
            minY = 10
            maxY = 30
            measure = '(C)'
        elif data_name_set == 'west_temp':
            data_names = ['base_west_temp','agent_west_temp']
            minY = 10
            maxY = 30
            measure = '(C)'
        elif data_name_set == 'elec_bill':
            data_names = ['base_bill', 'agent_bill', 'bill_difference']
            measure = '(억 원)'
        elif data_name_set == 'hvac_energy':
            data_names = ['base_hvac_energy', 'agent_hvac_energy', 'hvac_energy_difference']
            measure = '(kWh)'
            minY = -3000
            maxY = 10000
        elif data_name_set == 'hvac_energy_acc':
            data_names = ['base_hvac_energy_acc', 'agent_hvac_energy_acc', 'hvac_energy_difference_acc']
            measure = '(kWh)'
        else:
            minY = 0
            maxY = 0
            measure = None
                
        for name in data_names:
            agent.append({'name': name,'data': np.reshape(data[name].values, -1, 1).tolist()})

        if data_name_set == 'elec_bill':
            minY = 0
            maxY = max(max(agent[i]['data'] for i in range(len(agent))))
        if data_name_set == 'hvac_energy_acc':
            minY = 0
            maxY = max(max(agent[i]['data'] for i in range(len(agent))))

    else:
        agent = []
        hour = []
        month = []
        minY = 0
        maxY = 0
        measure = None

    trace_agents = []
    colors = ['#000000', '#ff0000', '#0000ff', '#00ff00', '#ffff00', '#ff00ff', '#00ffff', '#f0f0f0']
    for i in range(len(agent)):
        trace_agent = go.Scatter(
            x=list(range(len(agent[0]['data']))),
            y=agent[i]['data'],
            line=go.scatter.Line(
                width=1,
                color=colors[i]
            ),
            hoverinfo='y',
            name = DICT[agent[i]['name']],
        )
        trace_agents.append(trace_agent)


    # trace_energyplus = go.Scatter(
    #     x=list(range(len(energyplus))),
    #     y=energyplus,
    #     line=go.scatter.Line( 
    #         color='#99d6ff',
    #     ),
    #     hoverinfo='skip',
    #     name = 'energyplus',
    #     mode='lines'
    # )

    layout = go.Layout(
        height=450,
        xaxis=dict(
            # range=[0, 60],
            showgrid=False,
            showline=False,
            zeroline=False,
            fixedrange=True,
            tickvals=list(range(0, 8760, 730)),
            ticktext=month,
            title='Month',
        ),
        yaxis=dict(
            range=[min(0.5*minY, minY),
                max(0.5*maxY, maxY)],
            showline=False,
            fixedrange=True,
            zeroline=False,
            nticks=5,
            title=measure
        ),
        margin=go.layout.Margin(
            t=45,
            l=50,
            r=50
        )
    )
    return {'data':trace_agents, 'layout':layout}

@app.callback(
    Output(component_id='bill', component_property='children'), 
    [Input(component_id='observation-update', component_property='n_intervals')]
)
def update_bill(n_intervals):
    data = source.fetch(DB, COLLECTION).toDf()
    if not data.empty:
        hour = data['hour'].values[-1]
        month = hour // 730
        base_bill = data['base_bill'].values[-1]
        agent_bill = data['agent_bill'].values[-1]
        bill_difference = data['bill_difference'].values[-1]
        base_bill_text = html.Span('AI 미적용 전기료 : %.2f 억 원' % (base_bill), className='bill-span')
        agent_bill_text = html.Span('AI 적용 전기료 : %.2f 억 원' % (agent_bill), className='bill-span')
        bill_difference_text = html.Span('절감 전기료 : %.2f 억 원' % (bill_difference), className='bill-span')
        month_text = html.Span('경과시간 : %.0f 개월' % (month), className='bill-span')
        text = [base_bill_text,agent_bill_text,bill_difference_text, month_text]
    else:
        text = []

    # base_bill_text = '%.1f 억 원' % (base_bill / 100000000)
    # agent_bill_text = '%.1f 억 원' % (agent_bill / 100000000)
    # bill_difference_text = '%.1f 억 원' % (-bill_difference / 100000000)
    # text = html.Table(
    #     [
    #         html.Tr([html.Th(['AI 미적용 요금', 'AI 적용 요금', '전기료 절감'])]),
    #         html.Tr([html.Td([base_bill_text, agent_bill_text, bill_difference_text])])
    #     ],
    #     style=dict(align='center' backgroundColor='#1d8acc', color='white'))

    return text


if __name__=="__main__":
    app.run_server(
        debug=True,
        port=port
    )

