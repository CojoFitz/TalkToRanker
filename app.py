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
from helperFunctions import processHistogram,getBins,suggest_cutoff, checkAttribution, getFeatureList, prepareDF, fairnessCalc, suggestFairness
from ContextObject import ContextObject
from Response import Response
import warnings
from matchResponse import matchResponse
from dash_extensions import Keyboard

warnings.filterwarnings( #for some reason I get weird warnings with this version of pandas & plotly, but the new version breaks my graph 
    action="ignore",
    message=r"When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas\. Pass `\(name,\)` instead of `name` to silence this warning\.",
    category=FutureWarning,
    module=r"plotly\.express\._core",
)


arrowIcon = "https://cdn-icons-png.flaticon.com/512/3682/3682321.png"
loadingIcon = "https://media.tenor.com/On7kvXhzml4AAAAj/loading-gif.gif"


adm = 'admission_all.csv'
credit = 'credit_risk_all.csv'
df = pd.read_csv(r'datasets/'+credit)
target = 'y'
chatContext = ContextObject()
nonVisMatch = (1,6,8,9,10,11,13)
chatContext.visType = 0
defaultIds = df['id'].to_list()
chatContext.subsetId = defaultIds
chatContext.newId =defaultIds
filtered_df = df[df['id'].isin(chatContext.newId)]
previousMessage = 'NONE'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
custom_scrollbar_css = {
    'overflowY': 'scroll',
    'height': '75vh',
    'scrollbarWidth': 'thin',
    'scrollbarColor': '#cccccc #f0f0f0',
}

featureDict = getFeatureList(df)
allFeatures = ", ".join(featureDict['all'])
categorical_features = featureDict['categorical']
numerical_features = featureDict['numerical']

formatQuery = matchResponse(allFeatures)
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
    fig.update_layout(modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'resetScale', 'autoScale', 'toImage'])
    return fig


app.layout = html.Div([
    dbc.Container([
        html.H4("TalkToRanker", className="card-title", style={'text-align': 'center', 'padding' :'9px'}),
        html.Div(id="chat-output",
            style=custom_scrollbar_css,  # Apply custom scrollbar CSS
        ),
        html.Div([
            Keyboard(captureKeys=["Enter"], id="keyboard", n_keyups = 0),
            dbc.Input(id="user-input", type="text", placeholder="Type your message..."),
            html.Button( id="submit-button", n_clicks=0, style={"width": "30px", "height": "30px", "border":"none","background":"none"}, children=[html.Img(src="https://cdn-icons-png.flaticon.com/512/3682/3682321.png",style={"width": "30px", "height": "30px"}, id='submitIcon')]),
        ], style={"display": "flex", "align-items": "center", " flex-direction": "column-reverse","justify-content": "center"}),
        dcc.Store(id='chat-history', data=[]),
    ], className="p-5", style={"width": "40%", "margin-right": "auto", "margin-left": "0"}),
    dcc.Graph(id="plot", style={"width": "75vw", "height": "90vh"}, figure = blank_fig(), config= {'displaylogo': False}),
    html.Div(id='dummy-output', style={'display': 'none'})  # Dummy output component to trick callback function to work, don't remove


], className="d-flex justify-content-start", )




numMessages = 0





def genResponse(taskType, df, feature1):
    genResp = 'None'
    if(taskType == 0):
        genResp = 'Query not understood'
    elif(taskType == 2):
            cor = round(df[feature1+'_v'].corr(df[target]),3)
            genResp = ["The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'] #Correlation
    elif(taskType == 3):
        std = round(df[feature1+'_v'].std(),3)
        mean = round(df[feature1+'_v'].mean(),3)
        median = round(df[feature1+'_v'].median(),3)
        genResp = ['Information about feature: ' + feature1, 'Standard Deviation: ' + str(std), 'Mean: ' + str(mean), 'Median: ' + str(median)]
    elif(taskType == 4):
        genResp = 'Showing you the stability' #Stability Response

    elif(taskType == 5):
        mostImportant = [element.rstrip('_a') for element in (checkAttribution(df).tail(4)['feature'].tolist())]
        mostImportant.reverse()
        if len(mostImportant) > 1:
            joined_features = ', '.join(mostImportant[:-1]) + ', and ' + mostImportant[-1]
        else:
            joined_features = mostImportant[0]
        genResp = ['The four most important features are ', joined_features] #List most important features
    elif(taskType == 6):
        numerical_str = "numerical: " + ", ".join(numerical_features)
        categorical_str = "categorical: " + ", ".join(categorical_features)
        genResp = [numerical_str, categorical_str]
    elif(taskType == 7):
        genResp = ['Here are the fairness ratios:']
        genResp += ([f"{key}: {value}" for key, value in fairnessCalc(feature1, df).items()])
        
    return genResp
def updateGlobalFeatures(feature):
    if(feature not in showFeatures):
        showFeatures.pop()
        showFeatures.insert(0,feature)
    
def processQuery(userInput, contextObj):
    isTrackedVis = False
    trackedName = ''
    taskType = 0 #Zero will be our error state
    textResponse = []
    feature1 = processedInfo = num1 = num2 = None
    id = contextObj.newId
    #I will need to add the patterns for the other tasks
    parsedQuery = formatQuery.queryParser(userInput,contextObj.parsedInfo)
    matchedQuery = parsedQuery[0]


    if(parsedQuery[6]):
        textResponse = ['Intepreting query as: ' + parsedQuery[5]]
    if matchedQuery == 1:
        numEntries = '0'
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        mode = parsedQuery[1]
        num1 = parsedQuery[2]
        feature1 = parsedQuery[3]
        num2 = parsedQuery[4]
        idList = df[(df[feature1+'_v'] > float(num1)) & (df[feature1+'_v'] < float(num2))]['id'].tolist()
        if(mode == "Subset"):
            contextObj.subsetId = idList 
            numEntries = str(len(idList))
            contextObj.fairFilter = False
        else:
            id = np.intersect1d(idList, contextObj.subsetId).tolist()
            numEntries = str(len(id))
            if(feature1 == 'y'):
                contextObj.fairFilter = True
            else:
                contextObj.fairFilter = False


        textResponse.append('Applying ' + mode +', ' +  str(numEntries) + ' items matched')
    elif matchedQuery == 2 or matchedQuery == 3:
        taskType = matchedQuery
      #Example: "What is the correlation of the target with GRE_Score_v"
      #"What is the correlation of the target with TOEFL_Score_v"
        feature1 = parsedQuery[1]
        if(feature1+'_v' in df.columns):
            updateGlobalFeatures(feature1)
            subDf = df[df['id'].isin(chatContext.subsetId)]
            filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
           # cor = filtered_df[feature1+'_v'].corr(filtered_df[target])
           # textResponse = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'
            textResponse += (genResponse(taskType, filtered_df, feature1))
        else:
            textResponse.append("There is no feature with the name " + feature1 + ". Please check the spelling or verify that it is a feature.")
            feature1 = None
            taskType = -1


    elif matchedQuery == 4:
        taskType = 4
        textResponse.append(genResponse(taskType, None, None))
    elif matchedQuery == 5:
        taskType = 5 #Feature attribution stuff
        subDf = df[df['id'].isin(chatContext.subsetId)]
        filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
        textResponse.append(genResponse(taskType, filtered_df, None))
    elif matchedQuery == 6:
        taskType = contextObj.visType
        trackedName = parsedQuery[1]
        textResponse.append("Tracking message as: " + trackedName)
        chatContext.trackedResponses[trackedName] = len(contextObj.responses) - 1
    elif matchedQuery == 7:
        isTrackedVis = True
        trackedName = parsedQuery[1]
        if trackedName in chatContext.trackedResponses.keys():
            oldResponse = contextObj.responses[chatContext.trackedResponses[trackedName]]
            taskType = oldResponse.visType
            feature1 = oldResponse.oldParse[0]
            if(feature1):
                if(feature1 not in showFeatures):
                    showFeatures.pop()
                    showFeatures.insert(0, oldResponse.oldParse[0])
            subDf = df[df['id'].isin(chatContext.subsetId)]
            filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
            textResponse = textResponse + ["Old response is: " ]+ oldResponse.old + ['New Response is:'] + genResponse(taskType, filtered_df,feature1)
        else:
            taskType = -1
            textResponse =  textResponse + ['There is no tracked input named: ' + trackedName +', please check your spelling or enter a valid tracked response']
    elif matchedQuery == 8:
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        mode = parsedQuery[1]
        col = parsedQuery[2]
        categorical = parsedQuery[3]
        if(col in df.columns):
            numEntries = 0
            contextObj.fairFilter = False
            idList = df[df[col] == categorical]['id'].tolist()
            if(mode == "Subset"):
                contextObj.subsetId = idList 
                numEntries = str(len(idList))
            else:
                id = np.intersect1d(idList, contextObj.subsetId).tolist()
                numEntries = str(len(id))
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) + ' items matched')
        else:
            textResponse.append('Error, theere is no categorical feature: ' + col + ' please double check your query!')
    elif matchedQuery == 9:
        taskType = 6
        textResponse = genResponse(taskType, df, None)
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        #We might be better off having a system where it keeps in mind the last task type or something idk
    elif matchedQuery == 10:
        topNum = int(parsedQuery[1])
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        subDf = df[df['id'].isin(contextObj.subsetId)]
        id = subDf.nlargest(topNum, 'y')['id'].tolist()
        if(topNum > len(subDf)):
            textResponse.append('Selection of top ' + str(topNum) + ' is too big, selecting top ' + str(len(subDf)) + ' instead.')
            topNum = len(subDf)
        textResponse.append('Getting top ' + str(topNum) + ' items')
        suggestedCutoff = suggest_cutoff(topNum,subDf['y'].to_numpy())
        if(suggestedCutoff != None):
            textResponse += suggestedCutoff
        contextObj.fairFilter = True


    elif matchedQuery == 11:
        numEntries = '0'
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        mode = parsedQuery[1]
        feat = parsedQuery[2]
        a_columns = [col for col in df.columns if col.endswith('_a')]
        if(feat+'_a' not in a_columns):
            textResponse.append("There is no feature with the name " + feat + ". Please check the spelling or verify that it is a feature.")
        else:

            mask = df[feat+'_a'] == df[a_columns].max(axis=1)
            idList = df.loc[mask, 'id']
            if(mode == "Subset"):
                contextObj.subsetId = idList 
                numEntries = str(len(idList))
            else:
                id = np.intersect1d(idList, contextObj.subsetId).tolist()
                numEntries = str(len(id))
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) + ' items matched')
    elif matchedQuery == 12:
        feature1 = parsedQuery[1]
        taskType = 7
        if(contextObj.fairFilter):
            subDf = df[df['id'].isin(chatContext.subsetId)]
            subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in chatContext.newId else 'unselected')
            textResponse = genResponse(taskType,subDf,feature1)
        else:
            textResponse = ['Invalid Request. Fairness can only be calculated using filters involving the target.']
            taskType = 0
    elif matchedQuery == 13:
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        feature1 = parsedQuery[1]
        if(contextObj.fairFilter):
            subDf = df[df['id'].isin(chatContext.subsetId)]
            subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in chatContext.newId else 'unselected')
            textResponse = ['Suggested cutoff point is top '  + str(suggestFairness(feature1,subDf))]
        else:
            textResponse = ['Invalid Request. Fairness can only be calculated using filters involving the target.']

    if(taskType != 0): #This is to update context object given a successful parse
        parsedInfo = []
        oldResponse = "" 
        if((matchedQuery in nonVisMatch) and (len(contextObj.responses)>0)):
            oldResponse = contextObj.responses[len(contextObj.responses)-1].old
            parsedInfo = contextObj.parsedInfo
        else:
            parsedInfo =  [feature1,processedInfo,num1,num2,showFeatures]
            oldResponse = textResponse

        contextObj.newId = id
        contextObj.trackedVis = isTrackedVis
        contextObj.trackedName = trackedName
        contextObj.parsedInfo = parsedInfo
        contextObj.visType = taskType
        response = Response(id, oldResponse, textResponse, taskType, parsedInfo)
        contextObj.responses.append(response)
        return contextObj
    else:
        return None

@app.callback(
    [Output("chat-output", "children"),
     Output("user-input", "value"),  # Reset user input
     Output('chat-history', 'data'),
     ], 
    [Input("submit-button", "n_clicks"),
     Input("keyboard", "n_keyups"),
     ],
    [State("user-input", "value"),
     State('chat-history', 'data')],
     running=[(Output("user-input", "disabled"), True, False),
              (Output("keyboard", "disabled"), True, False),
              (Output("submit-button", "disabled"), True, False),
              (Output("submitIcon", "src"), loadingIcon, arrowIcon)],
)

def update_chat(n_clicks, n_keyups, user_input, chat_history):
    if (n_clicks > 0 or n_keyups > 0) and user_input:
        # Append the user's message to the chat history
        # Empty user input
        chat_history.append({'sender': 'user', 'message': [user_input]})

        currContext = processQuery(user_input,chatContext)
        if(currContext == None):
            chat_history.append({'sender': 'computer', 'message': ['Error or invalid input, please try again']})
        else:
            chat_history.append({'sender': 'computer', 'message': currContext.responses[len(currContext.responses)-1].new})
        user_input = ""

    chat_output = [
        dbc.Card(

 [
            dbc.CardBody(
                content, 
                className="user-bubble" if chat_history[i]['sender'] == 'user' else "computer-bubble",
                style={
                    "color": "white" if chat_history[i]['sender'] == 'user' else "black",
                      # top right bottom left
                      "padding": "4px 20px 4px 20px",

                }
            )
            for content in chat_history[i]['message']
        ],
            className="user-bubble-card" if chat_history[i]['sender'] == 'user' else "computer-bubble-card",
            style={
                "background-color": "#147efb" if chat_history[i]['sender'] == 'user' else "#D3D3D3",  # Background color
                "margin-right": "10px" if chat_history[i]['sender'] == 'user' else "auto",  # Align user on left, computer on right
                "margin-left": "10px" if chat_history[i]['sender'] != 'user' else "auto",  # Align computer on right, user on left
                "padding-top": "10px",
                "padding-bottom": "10px",
                "width": "200px",
                "margin-top": "10px" if i != 0 else "auto",

            }
        )
        for i in range(0,len(chat_history))
    ]
        
    return chat_output, user_input, chat_history

@app.callback(
    Output("dummy-output", "children"),  # Dummy output
    [Input("plot", "selectedData")]
)
def display_selected_data(selected_data):
    if selected_data:
        points = selected_data["points"]
        pointDf = pd.json_normalize(points)
        idList = pointDf['customdata'].tolist()
        flatIdList = [item for sublist in idList for item in sublist]
        chatContext.newId = flatIdList
        if(chatContext.visType == 4):
            chatContext.fairFilter = True
        else:
            chatContext.fairFilter = False
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
        unfilt_df = prepareDF(subDf, 'unselected',showFeatures)
        df_melted = prepareDF(filtered_df, 'selected',showFeatures)
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
                tracked = chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]]
                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                oldFilter = prepareDF(oldFilter,'selected',showFeatures)
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
                  "feature": [showFeatures[0] , showFeatures[1] , showFeatures[2], showFeatures[3]]}, color_discrete_sequence=['rgba(44,69,107, 0.2)','#0096FF'], )  
                fig.update_layout(modebar_remove=['select', 'lasso'])

            else:

                #this one is really weird, for some reason plotly does not have smooth animations for histograms
                #so I am pre-processing the data to be a barchart pretending to be a histogram

                tracked = chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]]
                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                featuresUsed =tracked.oldParse[4]
                oldMelted = prepareDF(oldFilter, 'selected',showFeatures)
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
                fig.update_layout(modebar_remove=['select', 'lasso'])

           



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
                oldFilter = subDf[subDf['id'].isin(chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]].oldId)]
                old_avgs = checkAttribution(oldFilter)
                old_avgs['oldNew'] = 'old'
                df_avgs['oldNew'] = 'new'
                combinedDf = pd.concat([old_avgs,df_avgs ], ignore_index=True)
                fig = px.bar(combinedDf, x="mean absolute attribution", y="feature", orientation='h', animation_frame="oldNew")
            fig.update_layout(modebar_remove=['select', 'lasso'])
        elif(chatContext.visType == 7):
            if(not chatContext.trackedVis):
                subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in chatContext.newId else 'unselected')
                featOne = chatContext.parsedInfo[0]
                ratioDf = pd.DataFrame(list(fairnessCalc(chatContext.parsedInfo[0],subDf).items()), columns=['category', 'ratio'])
                fig = px.bar(ratioDf, x="category", y="ratio", orientation='v')
                fig.update_yaxes(range=[0, 1])
            else:
                tracked = chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]]
                subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in chatContext.newId else 'unselected')

                newRatio = pd.DataFrame(list(fairnessCalc(chatContext.parsedInfo[0],subDf).items()), columns=['category', 'ratio'])
                newRatio['oldNew'] = 'new'

                subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in tracked.oldId else 'unselected')
                oldRatio = pd.DataFrame(list(fairnessCalc(chatContext.parsedInfo[0],subDf).items()), columns=['category', 'ratio'])
                oldRatio['oldNew'] = 'old'

                combinedDf = pd.concat([oldRatio,newRatio], ignore_index=True)
                fig = px.bar(combinedDf, x="category", y="ratio", orientation='v', animation_frame="oldNew")
                fig.update_yaxes(range=[0, 1])
            fig.update_layout(modebar_remove=['select', 'lasso'])

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
    app.run_server(debug=False)
