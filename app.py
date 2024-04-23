import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import json
import pandas as pd
import re
from ContextObject import ContextObject
from Response import Response
df = pd.read_csv('testCsv.csv')
chatContext = ContextObject()
chatContext.newId = df['id'].to_list()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
custom_scrollbar_css = {
    'overflowY': 'scroll',
    'height': '700px',
    'scrollbarWidth': 'thin',
    'scrollbarColor': '#cccccc #f0f0f0',
}

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig


app.layout = html.Div([
    dbc.Container([
        html.H4("User Input", className="card-title", style={'text-align': 'center', 'padding' :'9px'}),
        html.Div(id="chat-output",
            style=custom_scrollbar_css,  # Apply custom scrollbar CSS
        ),
        html.Div([
            dbc.Input(id="user-input", type="text", placeholder="Type your message..."),
            html.Img(src="https://cdn-icons-png.flaticon.com/512/3682/3682321.png", id="submit-button", n_clicks=0, style={"cursor": "pointer", "width": "30px", "height": "30px"}),
        ], style={"display": "flex", "align-items": "center", "justify-content": "center"}),
        dcc.Store(id='chat-history', data=[]),
    ], className="p-5", style={"width": "40%", "margin-right": "auto", "margin-left": "0"}),
    dcc.Graph(id="scatter-plot", style={"width": "900px", "height": "900px"}, figure = blank_fig()),
    html.Div(id="selected-points-output"),


], className="d-flex justify-content-start")


numMessages = 0

def processQuery(userInput, contextObj):
    target = 'y'
    taskType = 0 #Zero will be our error state
    textResponse = 'Query not understood'
    feature1 = feature2 = num1 = num2 = None
    print("Query is: " + userInput)
    id = contextObj.newId
    #I will need to add the patterns for the other tasks
    patternOne = r'Filter by (\d+)\s*<\s*([_A-Za-z]+)\s*<\s*(\d+)'
    patternTwo = r"What is the correlation of the target with (\w+)"
    patternThree = r"Show me the data"
    #30<feature<500

    match = re.match(patternOne, userInput)
    if match:
        taskType = 3 #This task is for filtering
        num1 = match.group(1)
        feature1 = match.group(2)
        num2 = match.group(3)
        #filtered_df = df[(df[feature1] > int(num1)) & (df[feature1] < int(num2))]
        id = df[(df[feature1] > float(num1)) & (df[feature1] < float(num2))]['id'].tolist()
        textResponse = str(len(id)) + ' students matched'
        

    match = re.match(patternTwo, userInput)
    if match:
        taskType = 2 #This task is for correlation between featue & y (need to adjust)

      #Example: "What is the correlation of the target with GRE_Score_v"
      #"What is the correlation of the target with TOEFL_Score_v"
        feature1 = match.group(1)
        cor = df[feature1].corr(df[target])
        textResponse = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'
  #This is the format ID will always be in. Since we will have the context of
  #the task type. The none values should not matter, as we can simply ignore
  #the un-needed indexes
    match = re.match(patternThree, userInput)
    if match:
        taskType = 3
        textResponse = 'Showing you the data'

    if(taskType != 0): #This is to update context object given a successful parse
        parsedInfo =  [feature1,feature2,num1,num2]
        contextObj.newId = id
        contextObj.parsedInfo = parsedInfo
        contextObj.visType = taskType
        response = Response(id, textResponse, textResponse, taskType, parsedInfo)
        contextObj.responses.append(response)
        return contextObj
    else:
        return None

@app.callback(
    [Output("chat-output", "children"),
     Output("user-input", "value"),  # Reset user input
     Output('chat-history', 'data'),
     ], 
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value"),
     State('chat-history', 'data')]
)


def update_chat(n_clicks, user_input, chat_history):
    if n_clicks > 0 and user_input:
        # Append the user's message to the chat history
        chat_history.append({'sender': 'user', 'message': user_input})
        currContext = processQuery(user_input,chatContext)
        if(currContext == None):
            chat_history.append({'sender': 'computer', 'message': 'Error or invalid input, please try again'})
        else:
            chat_history.append({'sender': 'computer', 'message':  currContext.responses[len(currContext.responses)-1].old})        
        # Empty user input
        user_input = ""

    chat_output = [
        dbc.Card(
            dbc.CardBody(
                message['message'], 
                className="user-bubble" if message['sender'] == 'user' else "computer-bubble",
                style={"color": "white"} if message['sender'] == 'user' else {"color": "black"}
            ),
            className="user-bubble-card" if message['sender'] == 'user' else "computer-bubble-card",
            style={
                "background-color": "#147efb" if message['sender'] == 'user' else "#D3D3D3",  # Background color
                "width": "200px" ,
                "margin-right": "10px" if message['sender'] == 'user' else "auto",  # Align user on left, computer on right
                "margin-left": "10px" if message['sender'] != 'user' else "auto",  # Align computer on right, user on left
            }
        )
        for message in chat_history
    ]
        
    return chat_output, user_input, chat_history


@app.callback(
    Output("selected-points-output", "children"),
    [Input("scatter-plot", "selectedData")]
)
def display_selected_data(selected_data):#This is for debugging purposes only, 
    if selected_data:
        points = selected_data["points"]
        pointDf = pd.json_normalize(points)
        print(pointDf['customdata'].tolist())
        return html.Ul([html.Li(f"X: {point['customdata']}") for point in points])
    else:
        return "No points selected."


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('chat-history', 'data')]
)

def update_chart(chat_history):
    if(len(chatContext.responses) > 0):
        filtered_df = df[df['id'].isin(chatContext.newId)]
        df_melted = pd.melt(filtered_df, id_vars=['y','id'], var_name='feature', value_name='value')
        if(chatContext.visType == 2):
            fig = px.scatter(df_melted, x="value", y="y",  facet_col="feature", facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  trendline="ols", category_orders={ 
                "feature": ['GRE_Score_v', 'SOP_v', 'TOEFL_Score_v', 'University_Rating_v']})
            fig.update_traces(customdata=df_melted['id'].astype(int))
        elif(chatContext.visType == 3):
            unfilt_df = pd.melt(df, id_vars=['y','id'], var_name='feature', value_name='value')
            df_melted['Source'] = 'filtered'
            unfilt_df['Source'] = 'unfiltered'
            result = pd.concat([df_melted,unfilt_df])
            fig = px.histogram(result, x="value",facet_col="feature", color="Source", facet_col_wrap=2, nbins=400, barmode="overlay", category_orders={ 
                "feature": ['GRE_Score_v', 'SOP_v', 'TOEFL_Score_v', 'University_Rating_v']}, color_discrete_sequence=['#00308F', '#00308F'])
            fig.update_layout(bargap=0.01)
            fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[0, None])
        fig.update_yaxes(range=[0, None])
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)

        return fig
    else:
        return blank_fig()

if __name__ == "__main__":
    app.run_server(debug=True)
