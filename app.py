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
from raceplotly.plots import barplot
from ContextObject import ContextObject
from Response import Response
import warnings

warnings.filterwarnings( #for some reason I get weird warnings with this version of pandas & plotly, but the new version breaks my graph 
    action="ignore",
    message=r"When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas\. Pass `\(name,\)` instead of `name` to silence this warning\.",
    category=FutureWarning,
    module=r"plotly\.express\._core",
)
adm = 'admission_all.csv'
credit = 'credit_risk_all.csv'
df = pd.read_csv(adm)
target = 'y'
chatContext = ContextObject()
chatContext.visType = 0
defaultIds = df['id'].to_list()
chatContext.subsetId = defaultIds
chatContext.newId =defaultIds
filtered_df = df[df['id'].isin(chatContext.newId)]
previousMessage = 'NONE'
trackedResponses = {}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
custom_scrollbar_css = {
    'overflowY': 'scroll',
    'height': '700px',
    'scrollbarWidth': 'thin',
    'scrollbarColor': '#cccccc #f0f0f0',
}
 #This is used to pull the index of the currently referenced tracked value, I need a more elegant way to do this

def checkAttribution(checkDf):
    checkDf = checkDf.loc[:, checkDf.columns.str.endswith('_a')]   
    checkDf = checkDf.abs()
    avgs = checkDf.mean(axis=0)
    df_avgs = pd.DataFrame(avgs).reset_index()
    df_avgs.columns = ['feature', 'mean absolute attribution']
    df_avgs = df_avgs.sort_values(by=['mean absolute attribution'], axis=0, ascending=True)
    return df_avgs
avgDf = checkAttribution(df)
topFeature = avgDf.tail(4)['feature'].tolist()
topFeature = [element.rstrip('_a') for element in topFeature]
topFeature.reverse()
showFeatures = topFeature

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
        ], style={"display": "flex", "align-items": "center", " flex-direction": "column-reverse","justify-content": "center"}),
        dcc.Store(id='chat-history', data=[]),
    ], className="p-5", style={"width": "40%", "margin-right": "auto", "margin-left": "0"}),
    dcc.Graph(id="plot", style={"width": "900px", "height": "900px"}, figure = blank_fig()),
    html.Div(id='dummy-output', style={'display': 'none'})  # Dummy output component to trick callback function to work, don't remove


], className="d-flex justify-content-start")




numMessages = 0

def prepareDF(processDf, selectionType): 
    #This formats the DF in a way that works best for the graph
    newDf = pd.melt(processDf, id_vars=['y','id'], var_name='feature', value_name='value')
    featuresUsed = [showFeatures[0] + '_v', showFeatures[1] + '_v', showFeatures[2] + '_v', showFeatures[3] + '_v']
    newDf = newDf[newDf['feature'].isin(featuresUsed)]
    newDf['Selection'] = selectionType
    return newDf


def processHistogram(df, feats, binEdges):
  #df is the df, feats is an array of features to be used, n_bin is num of bins

  hisDf = pd.DataFrame()
  for i in range((len(feats))):
    df_fil = df[df.feature == feats[i]+'_v']
    counts, bins = np.histogram(df_fil["value"], bins=binEdges[i])
    binnedDf = pd.DataFrame({"bins": bins[1:], "counts": counts})
    binnedDf['feature'] = feats[i]
    hisDf = pd.concat([hisDf, binnedDf], ignore_index=True)
  return hisDf
def getBins (df, feats):
    binEdges = []
    for i in feats:
        df_fil = df[df.feature == i + '_v']
        featEdge = np.histogram_bin_edges(df_fil['value'], bins='scott')
        binEdges.append(featEdge)
    return binEdges




def genResponse(taskType, df, feature1):
    genResp = 'None'
    if(taskType == 0):
        genResp = 'Query not understood'
    elif(taskType == 2):
            cor = df[feature1+'_v'].corr(df[target])
            genResp = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.' #Correlation
    elif(taskType == 3):
        genResp = 'Showing you the data' #Response to "Show me the data"
    elif(taskType == 4):
        genResp = 'Showing you the stability' #Stability Response
    elif(taskType == 5):
        genResp = 'Showing the most important features' #List most important features
    return genResp


def processQuery(userInput, contextObj):
    isTrackedVis = False
    trackedName = ''
    taskType = 0 #Zero will be our error state
    textResponse = ['Query not understood']
    feature1 = feature2 = num1 = num2 = None
    id = contextObj.newId
    #I will need to add the patterns for the other tasks
    patternOne = r'(Filter|Subset) by (\d+)\s*<\s*([_A-Za-z]+)\s*<\s*(\d+)' #Numerical
    patternTwo = r"What is the correlation of the target with (\w+)"
    patternThree = r"Show me the data"
    patternFour =  r"Show me the stability"
    patternFive =  r"What are the most important features"
    patternSix = r"Track the previous response as (\w+)"
    patternSeven = r"Show me tracked response (\w+)"
    patternEight = r"(Filter|Subset) the data when feature (\w+) is (\w+)" #categorical 


    match = re.match(patternOne, userInput)
    
    if match:
        numEntries = '0'
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        mode = match.group(1)
        num1 = match.group(2)
        feature1 = match.group(3)
        num2 = match.group(4)
        idList = df[(df[feature1+'_v'] > float(num1)) & (df[feature1+'_v'] < float(num2))]['id'].tolist()
        if(mode == "Subset"):
            contextObj.subsetId = idList #HEY CONOR THIS IS NOT DONE ALL OF THIS STUFF, FIX IT
            numEntries = str(len(idList))
        else:
            id = np.intersect1d(idList, contextObj.subsetId).tolist()
            numEntries = str(len(id))
        textResponse = ['Applying ' + mode +', ' +  str(numEntries) + ' items matched']
    match = re.match(patternTwo, userInput)
    if match:
        
        taskType = 2 
      #Example: "What is the correlation of the target with GRE_Score_v"
      #"What is the correlation of the target with TOEFL_Score_v"
        feature1 = match.group(1)
        if(feature1+'_v' in df.columns):
            if(feature1 not in showFeatures):
                    showFeatures.pop()
                    showFeatures.insert(0,feature1)
            filtered_df = df[df['id'].isin(chatContext.newId)]
           # cor = filtered_df[feature1+'_v'].corr(filtered_df[target])
           # textResponse = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'
            textResponse = [genResponse(taskType, filtered_df, feature1)]
        else:
            textResponse = ["There is no feature with the name " + feature1 + ". Please check the spelling or verify that it is a feature."]
            feature1 = None
            taskType = -1
    match = re.match(patternThree, userInput)
    if match:
        taskType = 3
        textResponse = [genResponse(taskType, None, None)]


    match = re.match(patternFour, userInput)
    if match:
        taskType = 4
        textResponse = [genResponse(taskType, None, None)]
    match = re.match(patternFive, userInput)
    if match:
        taskType = 5 #Feature attribution stuff
        textResponse = [genResponse(taskType, None, None)]
    match = re.match(patternSix, userInput)
    if match:
        taskType = contextObj.visType
        trackedName = match.group(1)
        textResponse = ["Tracking message as: " + trackedName]
        trackedResponses[trackedName] = len(contextObj.responses) - 1

    match = re.match(patternSeven, userInput)
    if match:
        isTrackedVis = True
        trackedName = match.group(1)
        if trackedName in trackedResponses.keys():
            oldResponse = contextObj.responses[trackedResponses[trackedName]]
            taskType = oldResponse.visType
            feature1 = oldResponse.oldParse[0]
            if(feature1):
                if(feature1 not in showFeatures):
                    showFeatures.pop()
                    showFeatures.insert(0, oldResponse.oldParse[0])
            filtered_df = df[df['id'].isin(chatContext.newId)]
            textResponse = ["Old response is: ", oldResponse.old[0],  'New Response is:',  genResponse(taskType, filtered_df,feature1)]
        else:
            taskType = -1
            textResponse = ['There is no tracked input named: ' + trackedName +', please check your spelling or enter a valid tracked response']
    match = re.match(patternEight, userInput)
    if match:
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        mode = match.group(1)
        col = match.group(2)
        categorical = match.group(3)
        if(col in df.columns):
            idList = df[df[col] == categorical]['id'].tolist()
            if(mode == "Subset"):
                contextObj.subsetId = idList
                contextObj.isSubset = True
            else:
                id = idList
            textResponse =['Applying ' + mode +', ' +  str(len(idList)) + ' items matched']
        else:
            textResponse = ['Error, theere is no categorical feature: ' + col + ' please double check your query!'] 

        
    if(taskType != 0): #This is to update context object given a successful parse
        parsedInfo =  [feature1,feature2,num1,num2,showFeatures]
        contextObj.newId = id
        contextObj.trackedVis = isTrackedVis
        contextObj.trackedName = trackedName
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
        chat_history.append({'sender': 'user', 'message': [user_input]})
        currContext = processQuery(user_input,chatContext)
        if(currContext == None):
            chat_history.append({'sender': 'computer', 'message': ['Error or invalid input, please try again']})
        else:
            chat_history.append({'sender': 'computer', 'message': currContext.responses[len(currContext.responses)-1].old})
        # Empty user input
        user_input = ""

    chat_output = [
        dbc.Card(

 [
            dbc.CardBody(
                content, 
                className="user-bubble" if message['sender'] == 'user' else "computer-bubble",
                style={"color": "white"} if message['sender'] == 'user' else {"color": "black"},
            )
            for content in message['message']
        ],
            className="user-bubble-card" if message['sender'] == 'user' else "computer-bubble-card",
            style={
                "background-color": "#147efb" if message['sender'] == 'user' else "#D3D3D3",  # Background color
                "width": "200px",
                "margin-right": "10px" if message['sender'] == 'user' else "auto",  # Align user on left, computer on right
                "margin-left": "10px" if message['sender'] != 'user' else "auto",  # Align computer on right, user on left
            }
        )
        for message in chat_history
    ]
        
    return chat_output, user_input, chat_history

@app.callback(
    Output("dummy-output", "children"),  # Dummy output
    [Input("plot", "selectedData")]
)
def display_selected_data(selected_data):#This is for debugging purposes only, 
    if selected_data:
        points = selected_data["points"]
        pointDf = pd.json_normalize(points)
        idList = pointDf['customdata'].tolist()
        flatIdList = [item for sublist in idList for item in sublist]
        chatContext.newId = flatIdList

        return flatIdList
    else:
        return "No points selected."


@app.callback(
    Output('plot', 'figure'),
    [Input('chat-history', 'data'),
     Input("dummy-output", "children")],
)

def update_chart(chat_history,flatIdList):
    #Instead of using df here, we will probably be better of pre-subsetting it. 
    if(len(chatContext.responses) > 0 ):
        if(chatContext.visType == -1):
            return blank_fig()
        subDf = df[df['id'].isin(chatContext.subsetId)]#subset df

        filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
        unfilt_df = prepareDF(subDf, 'unselected')
        df_melted = prepareDF(filtered_df, 'selected')
        result = pd.concat([df_melted,unfilt_df])
        
        if(chatContext.visType == 2):

            
                
            unfilt_df.loc[unfilt_df['id'].isin(chatContext.newId), 'Selection'] = 'selected'
            if(not chatContext.trackedVis):

                fig = px.scatter(unfilt_df, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={ 
                    "feature": [ showFeatures[0] + '_v', showFeatures[1] + '_v', showFeatures[2] + '_v', showFeatures[3] + '_v'], "Selection": ["selected","unselected"]})
            
                fig.update_yaxes(matches=None)
                trendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                    "feature": [showFeatures[0] + '_v', showFeatures[1] + '_v', showFeatures[2] + '_v', showFeatures[3] + '_v']})  
                
            else:
                tracked = chatContext.responses[trackedResponses[chatContext.trackedName]]
                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                oldFilter = prepareDF(oldFilter,'selected')
                featuresUsed = tracked.oldParse[4]
                
                fig = px.scatter(unfilt_df, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={ 
                    "feature": [ featuresUsed[0] + '_v', featuresUsed[1] + '_v', featuresUsed[2] + '_v', featuresUsed[3] + '_v'], "Selection": ["selected","unselected"]})
            
                fig.update_yaxes(matches=None)
                trendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                    "feature": [featuresUsed[0] + '_v', featuresUsed[1] + '_v', featuresUsed[2] + '_v', featuresUsed[3] + '_v']}) 

                trackTrend = px.scatter(oldFilter, x="value", facet_col_spacing=0.04, y="y", trendline_color_override="red", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                "feature": [featuresUsed[0] + '_v', featuresUsed[1] + '_v', featuresUsed[2] + '_v', featuresUsed[3] + '_v']})     
                trackTrend.update_traces(visible=False, selector=dict(mode="markers"))
                fig.add_traces(trackTrend.data)
            trendFig.update_traces(visible=False, selector=dict(mode="markers"))
            fig.add_traces(trendFig.data)
            fig.update_traces(selected=dict(marker=dict(color='#4590ff')), unselected=dict(marker=dict(color='rgba(44,69,107, 0.2)')))
            
        elif(chatContext.visType == 3):
            binEdge = getBins(unfilt_df,showFeatures)
            
            if(not chatContext.trackedVis):
                unfiltHis = processHistogram(unfilt_df, showFeatures, binEdge)
                newHis = processHistogram(df_melted, showFeatures, binEdge)
                newHis['Selection'] = 'Selected'
                unfiltHis['Selection'] = 'Unselected'
                newResult = pd.concat([unfiltHis,newHis])
                fig = px.bar(newResult, x="bins", y = "counts", facet_col="feature", facet_col_spacing=0.04, color="Selection", facet_col_wrap=2,  barmode="overlay", category_orders={ 
                  "feature": [showFeatures[0] , showFeatures[1] , showFeatures[2], showFeatures[3]]}, color_discrete_sequence=['rgba(44,69,107, 0.2)','#0096FF'])  
            else:

                #this one is really weird, for some reason plotly does not have smooth animations for histograms
                #so I am pre-processing the data to be a barchart pretending to be a histogram

                tracked = chatContext.responses[trackedResponses[chatContext.trackedName]]

                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                featuresUsed =tracked.oldParse[4]
                oldMelted = prepareDF(oldFilter, 'selected')
                unfiltHis = processHistogram(unfilt_df, featuresUsed, binEdge)
                unfiltHis['Selection'] = 'Unselected'
                newHis = processHistogram(df_melted, featuresUsed, binEdge)
                oldHis = processHistogram(oldMelted, featuresUsed, binEdge)
                unfiltHis['Selection'] = 'Unselected'
                oldHis['Selection'] = 'Selected'
                newHis['Selection'] = 'Selected'
                
                oldResult = pd.concat([unfiltHis,oldHis])
                newResult = pd.concat([unfiltHis,newHis])

                oldResult['oldNew'] = 'old'
                newResult['oldNew'] = 'new'
                


                completeDf = pd.concat([newResult,oldResult])

                graphRanges = []
                for i in featuresUsed:
                    rangeVal = completeDf[completeDf['feature'] == i]['counts'].max()

                    graphRanges.append(rangeVal)
                fig = px.bar(completeDf, x="bins", y = "counts", facet_col="feature", facet_col_spacing=0.04, facet_col_wrap=2, color="Selection", color_discrete_sequence=['rgba(44,69,107, 0.2)','#0096FF'], barmode="overlay", category_orders={ 
                    "feature": [featuresUsed[0] , featuresUsed[1] , featuresUsed[2] , featuresUsed[3]]},  animation_frame="oldNew", animation_group="bins")
                fig.update_yaxes(range= [0,graphRanges[0]], row=2, col = 1)
                fig.update_yaxes(range= [0,graphRanges[1]], row=2, col = 2)
                fig.update_yaxes(range= [0,graphRanges[2]], row=1, col = 1) #Bottom Left
                fig.update_yaxes(range= [0,graphRanges[3]], row=1, col = 2) #Bottom Right
        
            fig.update_layout(bargap=0.01)
           



        elif(chatContext.visType == 4):
            rankDf = subDf.copy() 
            rankDf['Selection'] = 'unselected'
            rankDf.loc[rankDf['id'].isin(chatContext.newId), 'Selection'] = 'selected'
            rankDf['rank'] = rankDf['y'].rank(ascending=False)
            fig = px.scatter(rankDf, x="rank",  y="y", color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], hover_data={'id': True},  category_orders={ 
                "Selection": ["selected","unselected"]})
            fig.update_traces(selected=dict(marker=dict(color='#4590ff')), unselected=dict(marker=dict(color='rgba(44,69,107, 0.2)')))
        elif(chatContext.visType == 5):

            df_avgs = checkAttribution(filtered_df)
            if(not chatContext.trackedVis):
                fig = px.bar(df_avgs, x="mean absolute attribution", y="feature", orientation='h')
            else:
                oldFilter = subDf[subDf['id'].isin(chatContext.responses[trackedResponses[chatContext.trackedName]].oldId)]
                old_avgs = checkAttribution(oldFilter)
                old_avgs['oldNew'] = 'old'
                df_avgs['oldNew'] = 'new'
                combinedDf = pd.concat([old_avgs,df_avgs ], ignore_index=True)
                fig = px.bar(combinedDf, x="mean absolute attribution", y="feature", orientation='h', animation_frame="oldNew")
        

        fig.update_layout(showlegend=False)
       # fig.update_xaxes(range=[0, None])
       # fig.update_yaxes(range=[0, None])S
       
        fig.update_yaxes(showticklabels=True, row=1)
        fig.update_yaxes(showticklabels=True, row=2)
        fig.update_xaxes(showticklabels=True, row=1)
        fig.update_xaxes(showticklabels=True, row=2)
        fig.update_yaxes(matches=None) 
        fig.update_xaxes(matches=None)

        return fig
    else:
        return blank_fig()

if __name__ == "__main__":
    app.run_server(debug=True)
