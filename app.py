import dash, math, warnings,json, jsonpickle, openai
from dash import dcc, html, callback, no_update, ctx
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from helperFunctions import processHistogram,getBins,suggest_cutoff, checkAttribution, getFeatureList, prepareDF, fairnessCalc, suggestFairness
from ContextObject import ContextObject
from Response import Response
from matchResponse import matchResponse
from dash_extensions import Keyboard, Mermaid
from creativeExplainer import creativeExplainer
from contextualPredictor import contextualPredictor



warnings.filterwarnings( #for some reason I get weird warnings with this version of pandas & plotly, but the new version breaks my graph 
    action="ignore",
    message=r"When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas\. Pass `\(name,\)` instead of `name` to silence this warning\.",
    category=FutureWarning,
    module=r"plotly\.express\._core",
)

arrowIcon = "https://cdn-icons-png.flaticon.com/512/3682/3682321.png"
loadingIcon = "https://media.tenor.com/On7kvXhzml4AAAAj/loading-gif.gif"
testImage = "https://placehold.co/600x400/EEE/31343C"
figureC = 'assets/figureC.png'
diagramImage = 'assets/DiagramOverview.png'
figureAb = 'assets/FigureBA.png'
good = 'assets/good.png'
error = 'assets/error.png'

ani1 = 'assets/Ani1.png'
ani2 = 'assets/Ani2.png'
ani3 = 'assets/Ani3.png'
ani4 = 'assets/Ani4.png'
ani5 = 'assets/Ani5.png'
ani6 = 'assets/Ani6.png'

adm = 'admission_all.csv'
credit = 'credit_risk_all.csv'
df = pd.read_csv(r'datasets/'+adm)
target = 'y'

nonVisMatch = (1,6,8,9,10,11,13,15)
#--------------------------

#------------------------------
previousMessage = 'NONE'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='TalkToRanker')
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





avgDf = checkAttribution(df)
topFeature = avgDf.tail(4)['feature'].tolist()
topFeature = [element.rstrip('_a') for element in topFeature]
topFeature.reverse()
showFeatures = topFeature

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_layout(modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'resetScale', 'autoScale', 'toImage'])
    return fig


def actionLine(message):
    return html.Div([
                    html.Img(src="https://cdn-icons-png.flaticon.com/512/3682/3682321.png",style={"width": "15px", "height": "15px",'display': 'inline-block', 'vertical-align': 'center'}),
                    html.P(message, style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '10px'})
                ])

def splitDisplay():
    return html.Div(
    style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},  # Flexbox to align items side by side and center them
    children=[
        html.Video(
        src='/assets/select.mp4',  # Path to your video file
        controls=False,  # Disable video controls
        autoPlay=True,  # Automatically start the video
        muted = True,
        loop=True,  # Loop the video
         style={'margin-left': '50px', 'margin-right': '50px', 'height':'50vh', 'width':'40vw'}),  # Image with some right margin
        html.Div(
            children=[
                html.H1("Try out the following messages and actions!"),
                actionLine('Show the correlation'),         
                actionLine('Show the correlation'),         
                actionLine('Show the correlation'),         
                actionLine('Show the correlation'),         
                actionLine('Show the correlation'),         

            ]
        ),
    ]
    )


test =     html.H5(
        'Your Header Text Here',
        style={
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',  # light grey with 50% opacity
            'padding': '10px',
            'textAlign': 'center'
        }
    )

carousel = dbc.Carousel(
    items=[
        {

 
            "key": "1",
            "src": ani1,
            "header": "User asks what factors are the most important for admissions",
                        "captionClassName" : "p-0 bg-dark border text-light bg-opacity-75 position-static"

        },
        {
            "key": "2",
            "src": ani2,
            "header": "User asks about the influence that GPA and TOEFL have on admittance",
                        "captionClassName" :"p-0 bg-dark border text-light bg-opacity-75 position-static"
        },
        {
            "key": "3",
            "src": ani3,
            "header": "User requests for the top 200 to be selected",
                        "captionClassName" :"p-0 bg-dark border text-light bg-opacity-75 position-static"
        },
                {
            "key": "4",
            "src": ani4,
            "header": "User asks how the GPA scores are spread out",
                        "captionClassName" :"p-0 bg-dark border text-light bg-opacity-75 position-static"
        },

        {
            "key": "5",
            "src": ani5,
            "header": "User compares tracked selection states",
                        "captionClassName" :"p-0 bg-dark border text-light bg-opacity-75 position-static"
        },
                {
            "key": "6",
            "src": ani6,
            "header": "User views the raw data as a table",
                        "captionClassName" :"p-0 bg-dark border text-light bg-opacity-75 position-static"
        }
    ],
        variant="dark",                style={
                    'margin-bottom': '20px',
                    'margin-left': '20%',
                    'margin-right': '20%',
                },
    indicators=False

)


modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Module", id = "modal-header")), #Have the module name here
                dbc.ModalBody("Text about module goes here", id = "modal-body"), #Have a dictionary generate the text based on module name
                dbc.ModalFooter(
                    html.Span(
                    id="close"
                    )
                ),
            ],
            id="modal",
            scrollable=True,
            is_open=False,
            size = "xl"
        ),
    ]
)




buttonsUsed = []


def tableTest():
    size = '500px'
    return html.Div([

    html.Table([
        # Table header
        html.Thead(
            html.Tr([
                html.Th("Gold Parse", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Regular expression", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Functionality", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
            ], style={'border': '1px solid black'})
        ),
        # Table body
        html.Tbody([
            html.Tr([
                html.Td("""(Filter or Subset) by {{number}}<{{feature}}<{{number}} """, style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""(Filter|Subset) by (-?\d+\.?\d*)<(\w+)<(-?\d+\.?\d*)""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("Filtering and subsetting by a numerical range", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'})
            ]),
            html.Tr([
                html.Td("""What is the correlation between {{feature}} and target""", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("""What is the correlation of the target with (\w+(?:\s*,\s*\w+)*)""", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("Correlation between feature and target", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'})
            ]),
            html.Tr([
                html.Td("What are the most important features", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("What are the most important features", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("Feature attribution", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'})
            ])
        ], style={'border': '1px solid black'})
    ], style={'borderCollapse': 'collapse', 'width': 'fit-content'})
])

def explainerTable():
    size = '500px'
    return html.Div([
    html.Table([
        # Table header
        html.Thead(
            html.Tr([
                html.Th("TasK Type ", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("User Input (Gold Parse Format)", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Static Response", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
            ], style={'border': '1px solid black'})
        ),
        # Table body
        html.Tbody([
            html.Tr([
                html.Td("""Correlation""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""What are the most important features?""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("The four most important features are feat1,feat2,feat3,feat4", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'})
            ]),
            html.Tr([
                html.Td("""Feature Attribution""", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("""What is the correlation of the target with (\w+(?:\s*,\s*\w+)*)""", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("Correlation between feature and target", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'})
            ]),
            html.Tr([
                html.Td("Distribution and feature information", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("What is the distribution of feat1", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Td("""Information about feature: feat1
                            Standard Dev: x
                            Mean: Y
                            Median: Z""", 
            style={'border': '1px solid black', 'width': size, 'textAlign': 'center'})
            ])
        ], style={'border': '1px solid black'})
    ], style={'borderCollapse': 'collapse', 'width': 'fit-content'})
])


def contextTable():
    size = '500px'
    return html.Div([
    html.Table([
        # Table header
        html.Thead(
            html.Tr([
                html.Th("With contextualUse", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Without contextualUse", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
            ], style={'border': '1px solid black'})
        ),
        # Table body
        html.Tbody([
            html.Tr([
                html.Td("""The correlation between the target (health status) and age is 0.826, indicating a strong positive relationship between age and the health outcome being assessed in the dataset.""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""The correlation between the target and age is 0.826, indicating a strong positive relationship between the two variables.""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
            ]),
        ], style={'border': '1px solid black'})
    ], style={'borderCollapse': 'collapse', 'width': 'fit-content'})
])


def selectTable():
    size = '500px'
    return html.Div([
    html.Table([
        # Table header
        html.Thead(
            html.Tr([
                html.Th("Query (gold parse format)", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Description", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
            ], style={'border': '1px solid black'})
        ),
        # Table body
        html.Tbody([
            html.Tr([
                html.Td("""Filter by numA<feat<numB""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""Filter by a numerical range based on a feature’s values""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
            ]),
            html.Tr([
                html.Td("""Select Top K""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""Select the top k items based on ranking""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
            ]),
            html.Tr([
                html.Td("""Filter the data such that feat is the most important feature""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""Select points in such a way that the attribution of feat makes it more or the most important.""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
            ]),
            html.Tr([
                html.Td("""Filter the data when feature feat is category""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
                html.Td("""Filter categorical features based on categories.""", style={'border': '1px solid black', 'width': '100px', 'textAlign': 'center'}),
            ]),
        ], style={'border': '1px solid black'})
    ], style={'borderCollapse': 'collapse', 'width': 'fit-content'})
])

def textSection(header, body):
    text = html.Div([
        html.H1(
            header,
            style={
                'text-align': 'center', 
                'margin-top': '40px'
            }
        ),
        *[
            html.Div(
                html.P(
                    paragraph,
                    style={
                        'font-size': '20px'
                    }
                ),
                style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'margin-bottom': '20px',
                    'margin-left': '20%',
                    'margin-right': '20%',
                }
            )
            for paragraph in body
        ]
    ])
    return text



def headerSubhead(header, body):
    return html.Div([
        html.H1(
            header,
            style={
                'text-align': 'center', 
                'margin-top': '40px'
            }
        ),
        html.H2(
            body,
            style={
                'text-align': 'center', 
                'font-style': 'italic', 
                'margin-top': '10px'
            }
        )
    ])
def modalFormatText(body):
    text = html.Div([
        *[
            html.Div(
                html.P(
                    paragraph,
                    style={
                        'font-size': '20px'
                    }
                ),
                style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'margin-bottom': '20px',
                    'margin-left': '3%',
                    'margin-right': '3%',
                }
            )
            for paragraph in body
        ]
    ])
    return text


def quoteBox(text_list):
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.P(paragraph, style={'margin': '0 0 10px 0'})  # Each paragraph
                    for paragraph in text_list
                ],
                style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'color': 'black',
                    'border': '1px solid black',
                    'padding': '10px',
                    'textAlign': 'center'
                }
            )
        ]
    )


def boldText(label, text):
    return html.P([html.B(label),text]
    )

def sources():
    return html.Div([
    html.P("[1] Brown, Tom, et al. \"Language models are few-shot learners.\" Advances in neural information processing systems 33 (2020): 1877-1901."),
    html.P("[2] Maddigan, Paula, and Teo Susnjak. \"Chat2VIS: generating data visualizations via natural language using ChatGPT, codex and GPT-3 large language models.\" Ieee Access 11 (2023): 45181-45193."),
    html.P("[3] Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, Kailong Wang, Yang Liu: “Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study”, 2023; arXiv:2305.13860."),
    html.P("[4] Slack, Dylan, et al. \"Explaining machine learning models with interactive natural language conversations using TalkToModel.\" Nature Machine Intelligence 5.8 (2023): 873-883."),
    html.P("[5] Nguyen, Van Bach, Jörg Schlötterer, and Christin Seifert. \"From black boxes to conversations: Incorporating xai in a conversational agent.\" World Conference on Explainable Artificial Intelligence. Cham: Springer Nature Switzerland, 2023."),
]),

def exampleQueries():
    return html.Div([ 
        html.P('To provide some inspiration for utilizing this interface, we have provided some query templates for you to use. You do not have to type these out verbatim in order for the interface to work. These serve as a baseline for what is possible in the interface, and good ways of formatting requests for specific interactions:'),
        dbc.Accordion(
        [
            dbc.AccordionItem(
                [
html.P("1. \"What is the distribution of {{features}}\" - Shows the distribution of the data."),
    
    html.P("2. \"Show me the stability\" - Shows stability of the data"),
    
    html.P("3. \"What are the most important features\" - Ranks feature attribution"),
    
    html.P("4. \"What is the correlation of the target with {{features}}\" - Shows correlation between a feature and the target"),
    
    html.P("5. \"(Filter or Subset) by {{number}}<{{feature}}<{{number}}\" - Filter OR subset based on a numerical feature range. Specify the upper and lower bound of the filter. Subsetting removes points not in the range, while filtering will just highlight the range."),
    
    html.P("6. \"(Filter or Subset) the data when {{feature}} is {{value}}\" - Filter OR subset based on the value of a categorical feature. Subsetting removes points not in the range, while filtering will just highlight the range."),
    
    html.P("7. \"Track the previous response as {{name}}\" - Tracks the previous response and stores the selected ids and subsets"),
    
    html.P("8. \"Show me tracked response {{name}}\" - Shows the tracked response, allowing the user to compare how the value handled by the response has changed in comparison to the original state."),
    
    html.P("9. \"What are the available features\" - Lists the features that are available"),
    
    html.P("10. \"Select the top {{number}}\" - Filters the selection to be the top n items. An example would be \"Select the top 8\""),
    
    html.P("11. \"(Filter or Subset) the data such that {{feature}} is the most important feature\" This creates a filter or subset to make sure that the columns selected are the most important feature."),
    
    html.P("12. \"Show me the fairness of the current filter on {{feature}}\" Shows the fairness of the given feature"),
    
    html.P("13. \"Suggest a better cutoff for fairness on {{feature}}\" Suggests a better selection for fairness of the given feature"),
    html.P("14. \"Show the rankings of the raw data\" Will display the raw data as a table with rankings")

                ],
                title="Click on me to see example queries",
            ),
        ]
        ),
    
          

])


introText = [""""""]
introHeader = 'Talk To Ranker: conversational interface for ranking-based decision-making'




overviewHeader = 'General Overview'
overviewBody = [
    """ Algorithmic rankers prove to be very useful in a multitude of different areas, as they assist greatly in the processes of making decisions. While those with data literacy may view ranker models as intuitive and easy to analyze, there is undeniably some level of knowledge required for one to obtain the answers to the questions they may have. As such, the average person may find it hard to utilize tools relating to rankings to the fullest extent. Our interface aims to make the process of utilizing rankers more accessible and intuitive. This is accomplished by leveraging, explainable artificial intelligence and visualizations to allow for rankers to be more accessible. Our interface is a hybrid conversational and visualization interface,  which generates visual and textual representations relevant to questions about the dataset. The interface is split into two views, the chat view (a) and visualization view (b). """,
    html.Img(src=figureAb),
    """As you can see in the chat view, the user is able to ask questions about a dataset pertaining to university admissions. After the user asks a question two things occur, a textual response is generated in the chat view as well as a visual response in the visualization view. Our interface is split into three major components, each with their own individual modules. They are that of the Explainable AI-Augmented Input, LLM-Augmented analytical module, and the Visualization Generator. Feel free to explore the different parts that make up these three components, with the following interactive diagram: """,
]

parserHeader = 'Parser'
parserText = [
"""The interactions in this interface primarily result from the user’s input in the chat-view. Since there are many different ways a user can ask the same question, it is important for the interface to be able to distinctly understand a question. As such, there needs to be a way to determine what the question is asking and what information is needed to answer it. It is important to have such information to accurately generate textual and visual explanations for the user. To help accomplish this, we are utilizing large language models such as GPT 3.5 [1] via API calls from OpenAi, to assist in this process. """,
"""One usage of LLMs in visualization interfaces is to generate executable code for visualizations. In an interface called Chat2Vis Paula and Susnjak demonstrated the capability of various LLMs to generate executable python code for the matplotlib library to create visualizations based on textual input [2]. This method was not selected to be used in our interface as there were some issues found with it. The code generated by the LLM was not always reliable, through testing this technique there were often times where the code would not execute or displayed wrong information. Additionally, with prompt engineering techniques that allow for instructions to be bypassed [3], a user can use this to inject whatever code they want. Furthermore, the unreliability and static nature of the visualizations drastically reduces the capability for interactivity to be integrated.""",
"""We took inspiration from similar works [4,5] creating a question bank containing formatted questions known as a gold parse. The gold parse is an idealized format for different question types to be in, which in turn user inputs can be matched to. The following is an example of some of the gold parses in our question bank:""",
tableTest(),
"""The interface works by checking if the user’s input matches to the gold parse, if it does not it will use an LLM such as GPT 3.5 [1] to format and match the user’s input to match a gold parse. For example if a user types “I am very curious to know the correlation between the target and age”, the LLM will re-format their input as "What is the correlation of age with target". After the input is made to be in the format of the gold parse, it will then be parsed using a list of regular expressions for different query types. Each query type has different parameters, such as features and numbers, of which will be stored and the type of query will be stored as a numerical value. At this point, the user’s input is matched with a gold parse and all of the information necessary to generate a response and visualization has been parsed. The information obtained from the parser is then subsequently passed off to the Context Predictor. """,
html.B('Sources'),
sources()]
exampleFeats = [f""" "age", "gender", "height", "weight", "blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate", "respiratory_rate", "temperature", "cholesterol_total" """]


contextHeader = 'Context Predictor'
contextBody = [
"""The Context Predictor is responsible for producing and storing contextual information that is necessary for the interface to function. The information that it stores is primarily the result of user interactions, this is information such as the most recent parse and points selected on a visualization. The Context Predictor’s primary function, as the name implies, is to make predictions about the contextual purpose of the dataset. There are different predictions that are generated by the Context Predictor, which are the contextualUse, contextualScore, and contextualElement. In order to generate these predictions, an LLM such as GPT 3.5 [1] is utilized.""",

""" The contextualUse is responsible for describing the dataset and its usage. The contextualUse is utilized by an LLM to ensure that it is generating contextually aware responses and information. The contextualUse is generated by providing the Context Predictor with a list of all the feature names in the dataset, and then asking the LLM to generate a description of what the dataset is for. An instance where this is useful can be highlighted with a dataset with the following feature names:""",
quoteBox(exampleFeats),
"""By looking at the features, it is quite easy to recognize that this dataset is based on medical information. The inclusion of features such as "blood_pressure_diastolic" and "heart_rate" help to distinguish the purpose of this dataset, as they are metrics that are commonly measured in medical environments. This dataset in particular highlights the need for the contextualUse, as the context of features such as “age” and “gender” can vary greatly depending on the dataset. """,
"""Since our interface expects datasets in a format with a column named ‘y’ for score, there is not always a clear understanding of its meaning. This is addressed by the contextualScore. The contextualScore is found by providing the contextualUse to an LLM, and asking it to generate a good replacement for y based on the contextualUse’ description. This obtained value will replace ‘y’, allowing for axis labels involving ‘y’ to be more informative.""",
"""The final area of prediction is the contextualElement. The contextualElement seeks to describe what is being referenced for each entry of the dataset. For example, if there was a dataset involving patient data, the contextualElement would be expected to identify that each entry or row refers to a patient. This process is done simply by providing an LLM with a list of all of the features in the dataset, and asking it to generate an appropriate element name. Once the element name is generated, it is utilized as a response to queries such as “select the top 100” where the expected response would be “showing the top 100 entries”; in this case it would replace element wioth the contextualElement.""",
html.B('Sources'),
f"""[1] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901."""





]


textualHeader = 'Textual Explainer'

textualInteraction = [f"""User:  “Hey, I am an admission counselor and I want to know if the correlation of the TOEFL_Score is important or not”""",
             f"""System: “Yes, the correlation between the target (likelihood of admission) and TOEFL Score is quite strong at 0.826, indicating that TOEFL Score is an important factor in predicting a student's likelihood of admission to the graduate program.”"""]


textualBody = ["""Our interface uses textual explanations generated with the assistance of LLMs such as GPT 3.5 [1] to provide rich textual responses. These textual responses are made using the textual explainer, whose function is as the name implies to provide explanations through text. The text explainer begins by receiving contextual information, such as filters, contextual use, and parsed information such as the task type and features. Once the textual explainer has the information it needs, it will generate what is called a static response based on the task type. The static response is essentially a template to generate a pre-made message that answers the question at its most basic level. The following table will show examples of the static response being generated based on some task types:""",
               explainerTable(),
               """The generation of the static response is a critical part for our process of generating text using a large language model. Since a large language model is being used to generate text, an obvious question is why even bother making a static response when one can directly send the user’s input to the large language model. The static response exists to remedy the limitations large language models have with handling mathematical calculations and large amounts of data. By having the static response, we can provide the LLM with a baseline of what an accurate response looks like, and allow the LLM to build off of it. These textual responses are generated using GPT 3.5 [1] via the OpenAI api, by giving the LLM a query using the user’s original input, the static response, and the contextualUse. The following is a complete example of an interaction between the user and the textual explainer:""",
               quoteBox(textualInteraction),
               """As can be observed in the interaction, the response was able to provide a mathematically based answer alongside an explanation for it. The user, who may not be as familiar with data science, can now know what the correlation is and why that may be important for their purposes. """,
               """In order for generated explanations to be contextually aware, the textual explainer is provided a string called the contextualUse by the Context Predictor. The contextualUse is a string that describes the dataset and its usage, this is then used to assist an LLM in making responses that are contextually aware. To generate the contextualUse, the Context Predictor is given a list of all the features in the dataset, and asks an LLM to generate a description of what the dataset is for. To demonstrate the effectiveness and need for this, let us assume we have a dataset with the following feature names:""",
                quoteBox(exampleFeats),
                """By looking at the features, it is quite easy to recognize that this dataset is based on medical information. The inclusion of features such as "blood_pressure_diastolic" and "heart_rate" help to distinguish the purpose of this dataset, as they are metrics that are commonly measured in medical environments. This dataset in particular highlights the need for the contextualUse, as the context of features such as “age” and “gender” can vary greatly depending on the dataset. To help highlight the need for the contextualUse, we included an example of an identical question being asked with and without the contextualUse:""",
                html.Div([ boldText('Question:','What is the correlation of the target with age?'),
                boldText('Static Response:','The correlation between the target and age is 0.826'),
                boldText('Contextual Use (Generated via GPT 3.5[1]):', 'This dataset is likely used for monitoring and analyzing the health status of individuals, including assessing cardiovascular risk factors and overall well-being.'),
                                contextTable()
                ]),
                """As evident in the two responses generated above, responses generated with the contextualUse offer much more informative and relevant responses to the dataset as a whole. This also allows for responses to provide more clarification. Most importantly, this serves to make the interface more accessible to differing levels of data-literacy.""",
                html.B('Sources'),
f"""[1] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901."""]
visualHeader = 'Visual Explainer'
visContextUse = ["""This dataset is likely used to predict or analyze graduate school admissions decisions based on applicants' academic scores, university ranking, statement of purpose, letters of recommendation, GPA, and research experience."""]

visualBody = [f"""Just as textual explanations are important to decision making, visualizations play an equally important role in the process. Our interface recognizes the importance of visualizations through the inclusion of the visual explainer, which aims to provide rich visualizations based on user input. These visualizations were generated through the use of dash and plotly, which offer native animation and selection tools.""",
            f"""The visual explainer generates visualizations based on user interactions, such as making selections on the graph or asking questions in the chat. Chat-based interactions are handled by the parser, which matches user responses to a templated format called a gold parse. Each gold parse has a unique task type associated with it, these task types determine what visualization is to be shown. For example, questions that involve correlation will show a scatter plot; whereas questions that ask about distribution will show a histogram. As for other interactions, user selections through a lasso or box select, can also update the visualization on what areas are highlighted.""",

              f"""Since the interface is designed for rankers, visualizations that support scatter plots will give information about the ranks of points. The information is given based on coloring, wherein the higher ranked points are given darker colors and the lower ranked points are given lighter colors. This can be shown in figure c.): """,
                  html.Img(src=figureC),
                f"""Selections are another area of interactivity supported by our interface, and they allow for users to have more control over the data. Selections can be done by using a lasso and box select tool on the points of the graph you wish to see. Alternatively, a user can also request to perform a selection through the chat. There are two types of selections that can be done as well, filtering and subsetting. Filtering will highlight the points that are selected and influence textual responses, however, the unselected points will remain present on the visualization with a grayed out color. Subsetting on the other hand allows for points to entirely be removed from the visualization as a whole, subsetting will also impact textual responses based on the group subsetted. These two functionalities are useful, as it allows users to cut out information not relevant or keep track of information for specific groups.""",
                f"""Since selecting with a lasso select tool isn’t always the most precise, textual selections prove to be useful for cases where a user may want a specific group selected. Here are the supported textual selection types in our interface, assume that all of these selections will be used for filtering:""",
                selectTable(),
                f"""Every entry in the dataset is expected to have a unique numerical ID assigned to it. The Context Predictor maintains a global record of the IDs, allowing for selections to remain persistent amongst visualizations. That means a user can perform a selection on a graph that shows the ranker's stability, and see how the selection they made impacted the distribution. This also applies for visualizations where there are more than one feature in the visualization, if one subplot in a view is selected, then the selections made will be reflect on the other plots.""",
                f"""As users begin to update filters a lot, they may also be interested in knowing the impact of their filters. This is addressed through feature tracking, which allows users to track responses. For example, let’s say a user wishes to keep track of the current state of selections for a scatter plot graph that shows correlation. They can request in the chat to “track the response as trackedName”, this will in turn keep track of what the previous response was alongside the visualization state of the response. As the user updates their selections, they can eventually ask to see the tracked response. Upon requesting the tracked response, they will be shown a visualization comparing the old from the new, alongside a chat message comparing the old and new evaluation generated.""",
                f"""The visual explainer makes use of the Context Predictor to enhance its descriptiveness. Since the format of datasets in our interface expects a column named ‘y’ for score, there is not always a clear understanding of its meaning. This is addressed by the Context Predictor’s contextScore. The contextScore is found by providing the contextualUse, which is a generated description of the dataset, to an LLM such as GPT 3.5[1]. Once the LLM has the contextualUse, they will be asked to generate a more descriptive name to replace ‘y’.This process is initialized at the start, so this name will be consistently maintained throughout a session.  Let us see an example where a visualization was generated showing the correlation between the TOEFL_Score and the target(y):""",
                  html.Img(src=figureC),
                html.Div([
                    html.P('As seen above, using the Context Predictor, the LLM was able to identify that the Y score likely refers to the admission decisions with this dataset. The contextualUse here was:'),
                    quoteBox(visContextUse),
                    html.P("""The contextualUse in this scenario helped to identify a suitable name for the value of y, that is informative and practical for the purposes of the user. """)
                    ]),
                    html.B('Sources'),
f"""[1] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901."""
            ]

featureAttrBody = ["""Machine learning models, like algorithmic rankers, are often considered black boxes because their internal workings are difficult to interpret. To address this, Explainable AI (XAI) methods provide ways to explain the outcomes of such models. In this work, we utilize SHAP[1], a model-agnostic, post-hoc XAI method, to explain the results of the rankers. Using the official SHAP package [https://github.com/shap/shap], we generate feature attribution explanations for each data item during pre-processing and load this data on the interface. For every data item, feature attribution is represented as a positive or negative score for each feature, reflecting the feature's impact on the model's outcome—in this case, the ranking score. Users can query the feature attributions for a custom group of data items to determine which features are most significant for that group. The backend of the interface calculates the average of the absolute feature attributions for each feature within the group.""",
                   html.B('Sources'),
                   """[1]Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017)"""
]


    
app.layout = html.Div([
    dcc.Store(id='chat-context', storage_type='memory'),


html.Div(
    children=[
    headerSubhead('Talk To Ranker','A conversational interface for ranking-based decision-making'),
    textSection('',introText+overviewBody),
    html.H1(
            "Click on each yellow module to learn more!",
            style={
                'text-align': 'center', 'margin-top' : '40px'}
        ),
        modal,
        cyto.Cytoscape(
        id='cytoscape-flowchart',
        layout={'name': 'preset',
                },
        style={'width': '95vw', 'height': '500px'},
        elements=[
           # {'data': {'id': 'VizViewLabel', 'label': 'Visualization View'}, 'position': {'x': 200, 'y': 325},'classes': 'unclick whiteNode'},
            
            { 'data': { 'id': "Feature", 'label': "Feature Attributions", 'group': "nodes", 'parent': "p1" } , 'position': {'x': 0, 'y': 200}},
            { 'data': { 'id': "Data", 'label': "Data", 'group': "nodes", 'parent': "p1" }, 'position': {'x': 0, 'y': 100} ,'classes': 'unclick whiteNode'},
            { 'data': { 'id': "p1", 'label': "Explainable AI-Augmented Input", 'group': "nodes" },'classes': 'unclick grouperNode bottom'},

            { 'data': { 'id': "Context", 'label': "Context Predictor", 'group': "nodes", 'parent': "p2" } , 'position': {'x': 150, 'y': 200}},
            { 'data': { 'id': "Text", 'label': "Textual Explainer", 'group': "nodes", 'parent': "p2" } , 'position': {'x': 150, 'y': 150}},
            { 'data': { 'id': "Parser", 'label': "Parser", 'group': "nodes", 'parent': "p2" }, 'position': {'x': 150, 'y': 100} },
            { 'data': { 'id': "p2", 'label': "LLM-Augmented analytical module", 'group': "nodes" },'classes': 'unclick grouperNode bottom'},


            { 'data': { 'id': "Chat", 'label': "Text Chat", 'group': "nodes", 'parent': "p3" } , 'position': {'x': 300, 'y': 200}, 'classes': 'unclick whiteNode'} ,
            { 'data': { 'id': "Vis", 'label': "Visual Explainer", 'group': "nodes", 'parent': "p3" }, 'position': {'x': 300, 'y': 100} },


            { 'data': { 'id': "p3", 'label': "Visualization Generator", 'group': "nodes" },'classes': 'unclick grouperNode bottom'},
            {'data': {'source': 'p3', 'target': 'p2', 'label' : 'Adapt Response'},'classes': 'double top'},
            {'data': {'source': 'p3', 'target': 'p2', 'label' : 'Generate'},'classes': 'double bottom'},
            {'data': {'source': 'p1', 'target': 'p2', 'label' : 'Process'},'classes': 'top'},






        ],
                        userZoomingEnabled=False,
                        userPanningEnabled=False,
                        autolock=True,
                        stylesheet=[
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-halign':'center',
                'text-valign':'center',
                'font-size': '10px',
                'width':'58px',
                'height':'40px',
                'shape':'square',
                'text-max-width' : '70px',
                'text-wrap':'wrap',
                'background-color': '#feff9c',
                'border-width' : '1px',
                'border-color' : 'black'

            }
        },
                    {
                'selector': 'edge',
                'style': {
                    'content': 'data(label)',
                    'curve-style': 'straight',
                    'font-size': '10x',
                    'text-max-width' : '60px',
                                    'text-wrap':'wrap',
                    
                    'line-color': 'black',
                                        'target-arrow-color': 'black',
                    'target-arrow-shape': 'triangle',
                }
            },  
              {
                'selector': '.offsetTop',
                'style': {
                    'target-endpoint': '39px -30px',
                    'source-endpoint': '-39px -30px',
                    'source-arrow-color': 'black',
                    'source-arrow-shape': 'triangle-tee',
                    'source-arrow-width': '400px',

                }
            },  

                          {
                'selector': '.offsetBottom',
                'style': {
                    'source-endpoint': '0 30px',
                         'target-endpoint': '0 30px'
                }
            },  
            {
                'selector': '.double',
                'style': {
                                        'curve-style': 'straight',

                    'target-arrow-color': 'black',
                    'target-arrow-shape': 'triangle',
                    'line-color': 'black',
                    'source-arrow-color': 'black',
                    'source-arrow-shape': 'triangle',
                }
            },
            {
                'selector': '.top',
                'style': {
                       'text-margin-y' : '-15px'
                }
            },
      {                      'selector': '.bottom',
                'style': {
                       'text-margin-y' : '12px'
                },},
            {
                'selector': '.leftSide',
                'style': {
                       'text-margin-x' : '-80px'
                }
            },            {
                'selector': '.softRight',
                'style': {
                       'text-margin-x' : '60px'
                }
            }, {
                'selector': '.softLeft',
                'style': {
                       'text-margin-x' : '-50px'
                }
            },
                        {
                'selector': '.rightSide',
                'style': {
                       'text-margin-x' : '70px'
                }
            },
            {
                'selector': '.unclick',
                'style': {
                       'events' : 'no'
                }
            },

            {
                'selector': '.grouperNode',
                'style': {
                    'shape':'roundrectangle',
                'background-color': '#eeeeee',
                'text-halign':'center',
                'text-valign':'bottom',
                }
            },
                    {
                'selector': '.whiteNode',
                'style': {
                'background-color': '#d3d3d3',
                }
            },
            
        ]


    ),
    textSection('Example interactions', ['The following  gallery shows off just some of the numerous interactions that our interface is capable of handling. Feel free to check them out for yourself here: ']),
    carousel,
    #textSection('Example Queries',[exampleQueries()]),
    #textSection('Sources',[sources()]),

    textSection('Try the interface out for yourself: ',''),
    textSection('',''),


    

],
),
html.Div(["Enter OpenAi Api Key to start:",
            html.Div(dcc.Input(id='apikey', type='password')),
            html.Button('Submit', id='apiButton'),
            html.Div(id='output-container-button', children=[
        html.Img(id='apiIcon', src=error,style={'width': '30px', 'height': '30px', 'margin-left' : '10px'})
    ])
        ],style={"display": "flex", "align-items": "center", " flex-direction": "column-reverse","justify-content": "center"}),

    
    
    
    html.Div([
    dbc.Container([
        html.H4("TalkToRanker", className="card-title", style={'text-align': 'center', 'padding' :'9px'}),
        html.Div(id="chat-output",
            style=custom_scrollbar_css,  # Apply custom scrollbar CSS
        ),
        html.Div([
            Keyboard(
                id="keyboard",
                captureKeys=["Enter"],
                n_keyups=0,
                children=[
                    dbc.Input(id="user-input", type="text", placeholder="Type your message...", disabled=True)
                ]
            , style={'width':'100%'}),
            html.Button( id="submit-button", n_clicks=0, style={"width": "30px", "height": "30px", "border":"none","background":"none"}, children=[html.Img(src="https://cdn-icons-png.flaticon.com/512/3682/3682321.png",style={"width": "30px", "height": "30px"}, id='submitIcon')]),
        ], style={"display": "flex", "align-items": "center", " flex-direction": "column-reverse","justify-content": "center"}),
        dcc.Store(id='chat-history', data=[]),
        dcc.Store(id='features', storage_type='memory', data=[]),

    ], className="p-5", style={"width": "40%", "margin-right": "auto", "margin-left": "0"}),
    dbc.Stack(
    [
      dcc.Loading(id = "loading-icon", 
                children=[
        dcc.Graph(id="plot", style={"width": "65vw", "height": "90vh"}, figure = blank_fig(), config= {'displaylogo': False})],type="default"),]

    ),
    html.Div(id='dummy-output', style={'display': 'none'}),  # Dummy output component to trick callback function to work, don't remove

], className="d-flex justify-content-start", )])


@callback(
   [Output('chat-context', 'data',allow_duplicate=True),
    Output('apiIcon','src'),
    Output('user-input', 'disabled'),
    Output("submit-button", "disabled")
    ],
    Input('apiButton', 'n_clicks'),
    State('apikey', 'value'),
    running=[Output("apiIcon", "src"), loadingIcon, error],
    prevent_initial_call=True
)



def createContext(n_clicks, value):
    if(value is not None):
        apiKey = value
        validKey = check_openai_api_key(apiKey)
        if(validKey):
            icon = good
            chatContext = ContextObject()
            chatContext.visType = 0
            defaultIds = df['id'].to_list()
            chatContext.subsetId = defaultIds
            chatContext.newId =defaultIds
            chatContext.apiKey = apiKey
            predictedContext = contextualPredictor(allFeatures, apiKey)
            contextualUsage = predictedContext.predictContextualUse()
            scoreName = predictedContext.contextualizeScore(contextualUsage)
            elementName = predictedContext.contextualizeElement()
            chatContext.elementName = elementName
            chatContext.scoreName = scoreName
            chatContext.contextUse = contextualUsage
            pickledChatContext = jsonpickle.encode(chatContext)
            return pickledChatContext,icon,False,False
    return None, error, True, True

textDict ={
    "Context Predictor" : modalFormatText(contextBody),
    'Visual Explainer' : modalFormatText(visualBody),
    'Parser' : modalFormatText(parserText),
    'Textual Explainer' : modalFormatText(textualBody),
    'Feature Attributions' : modalFormatText(featureAttrBody),
}
        
@app.callback(
   [ Output("modal", "is_open"),
    Output("modal-header", "children"),
    Output('cytoscape-flowchart', 'tapNodeData'),
    Output('modal-body','children')
    
    ],
    [Input('cytoscape-flowchart', 'tapNodeData'), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open,n1['label'], None, textDict[n1['label']]
    return is_open,n1, None, None

numMessages = 0




def genResponse(taskType, df, feature1,userInput,chatContext):
    #We can have it take chatContext as a parameter
    explainer = creativeExplainer(chatContext.contextUse, chatContext.apiKey)
    genResp = 'None'
    staticResp = 'None'
    if(taskType == 0):
        genResp = 'Query not understood'
    elif(taskType == 2):
        genResp = []
        staticResp = []
        for feat in feature1:
            cor = round(df[feat+'_v'].corr(df[target]),3)
            staticResp += ["The correlation between the target and " + feat + ' is ' + str(cor)+'.'] #Correlation
        
        staticInput = ' '.join(staticResp)
        genResp = [explainer.explainGen(userInput,staticInput)]

    elif(taskType == 3):
        genResp = []
        staticResp = []
        for feat in feature1:
            std = round(df[feat+'_v'].std(),3)
            mean = round(df[feat+'_v'].mean(),3)
            median = round(df[feat+'_v'].median(),3)
            staticResp += ['Information about feature: ' + feat, 'Standard Deviation: ' + str(std), 'Mean: ' + str(mean), 'Median: ' + str(median)]
        staticInput = ' '.join(staticResp)
        genResp = [explainer.explainGen(userInput,staticInput)]

    elif(taskType == 4):
        genResp = 'Displaying the stability (NOTE DO NOT ANSWER ABOUT HOW STABLE THE DATASET IS, AS YOU DO NOT HAVE THAT INFORMATION GIVEN TO YOU. You can answer supplementary questions, but nothing specific to the dataset. If you do not know how to respond, just say you are displaying the stability)' #Stability Response
        genResp = explainer.explainGen(userInput,genResp)

    elif(taskType == 5):
        mostImportant = [element.rstrip('_a') for element in (checkAttribution(df).tail(4)['feature'].tolist())]
        mostImportant.reverse()
        if len(mostImportant) > 1:
            joined_features = ', '.join(mostImportant[:-1]) + ', and ' + mostImportant[-1]
        else:
            joined_features = mostImportant[0]
        staticResp = ['The four most important features are ', joined_features] #List most important features
        staticInput = ' '.join(staticResp)
        genResp = explainer.explainGen(userInput,staticInput)

    elif(taskType == 6):
        numerical_str = "numerical: " + ", ".join(numerical_features)
        categorical_str = "categorical: " + ", ".join(categorical_features)
        genResp = [numerical_str, categorical_str]
    elif(taskType == 7):
        genResp = ['Here are the fairness ratios:']
        genResp += ([f"{key}: {value}" for key, value in fairnessCalc(feature1, df).items()])
    elif (taskType == -1):
        staticInput = 'There is no statically generated response for this question. Please make your best determination onm if this question should be answered or not. If you are not able to answer the question, be sure to say it. Otherwise, answer the question normally.'
        genResp = explainer.explainGen(userInput,staticInput)
        
    return [genResp, staticResp]




def processQuery(userInput, contextObj):
    #MARKER 2
    isTrackedVis = False
    staticResp = ''
    trackedName = ''
    taskType = -1 #Zero will be our error state
    textResponse = []
    feature1 = processedInfo = num1 = num2 = None
    id = contextObj.newId
    #I will need to add the patterns for the other tasks
    formatQuery = matchResponse(allFeatures,contextObj.apiKey)
    if(len(contextObj.responses) <= 0):
        parsedQuery = formatQuery.queryParser(userInput,contextObj.parsedInfo, 'No messages sent yet', 'No messages sent yet')
    else:
        parsedQuery = formatQuery.queryParser(userInput,contextObj.parsedInfo,contextObj.responses[len(contextObj.responses) - 1].old,contextObj.responses[len(contextObj.responses) - 1].userQuestion)

    matchedQuery = parsedQuery[0]

    if(parsedQuery[6]):
        textResponse = []
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

        if(feature1 not in featureDict['all']):
            feature1 = matchResponse(allFeatures,contextObj.apiKey).reMatchFeature(feature1, contextObj.scoreName)
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


        textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ contextObj.elementName +' matched')
    elif matchedQuery == 2 or matchedQuery == 3:
        taskType = matchedQuery
      #Example: "What is the correlation of the target with GRE_Score_v"
      #"What is the correlation of the target with TOEFL_Score_v"
        feature1 = parsedQuery[1]
        for i in range(len(feature1)):
            if(feature1[i] not in featureDict['all']):
                feature1[i] = matchResponse(allFeatures,contextObj.apiKey).reMatchFeature(feature1[i],contextObj.scoreName)
        if(set(feature1).issubset((set(featureDict['all'])))):
            contextObj.features = feature1
            subDf = df[df['id'].isin(contextObj.subsetId)]
            filtered_df = subDf[subDf['id'].isin(contextObj.newId)]
           # cor = filtered_df[feature1+'_v'].corr(filtered_df[target])
           # textResponse = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'
            genResp = genResponse(taskType, filtered_df, feature1, userInput,contextObj)
            staticResp = genResp[1]
            textResponse += (genResp[0])
        else:
            textResponse.append("One of your entered features has an error"  + ". Please check the spelling or verify that it is a feature.")
            feature1 = None
            taskType = -1


    elif matchedQuery == 4:
        taskType = 4
        genResp = genResponse(taskType, None, None, userInput,contextObj)
        staticResp = genResp[1]
        textResponse.append(genResp[0])
    elif matchedQuery == 5:
        taskType = 5 #Feature attribution stuff
        subDf = df[df['id'].isin(contextObj.subsetId)]
        filtered_df = subDf[subDf['id'].isin(contextObj.newId)]
        genResp = genResponse(taskType, filtered_df, None,userInput,contextObj)
        staticResp = genResp[1]
        textResponse.append(genResp[0])
    elif matchedQuery == 6:
        taskType = contextObj.visType
        trackedName = parsedQuery[1]
        textResponse.append("Tracking message as: " + trackedName)
        contextObj.trackedResponses[trackedName] = len(contextObj.responses) - 1
    elif matchedQuery == 7:
        isTrackedVis = True
        trackedName = parsedQuery[1]
        if trackedName in contextObj.trackedResponses.keys():
            oldResponse = contextObj.responses[contextObj.trackedResponses[trackedName]]
            taskType = oldResponse.visType
            feature1 = oldResponse.oldParse[0]
            if(feature1):
                contextObj.features = feature1
            subDf = df[df['id'].isin(contextObj.subsetId)]
            filtered_df = subDf[subDf['id'].isin(contextObj.newId)]
            textResponse = textResponse + ["Old evaluation is: " ]+ oldResponse.oldStatic + ['New evaluation is:'] + genResponse(taskType, filtered_df,feature1,userInput,contextObj)[1]
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
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ contextObj.elementName +' matched')
        else:
            textResponse.append('Error, theere is no categorical feature: ' + col + ' please double check your query!')
    elif matchedQuery == 9:
        taskType = 6
        textResponse = genResponse(taskType, df, None,userInput,contextObj)[0]
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
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
        textResponse.append('Getting top ' + str(topNum) + ' ' + contextObj.elementName)
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
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ contextObj.elementName +' matched')
    elif matchedQuery == 12:
        feature1 = parsedQuery[1]
        taskType = 7
        if(contextObj.fairFilter):
            subDf = df[df['id'].isin(contextObj.subsetId)]
            subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in contextObj.newId else 'unselected')
            textResponse = genResponse(taskType,subDf,feature1,userInput,contextObj)[0]
        else:
            textResponse = ['Invalid Request. Fairness can only be calculated using filters involving the target.']
            taskType = -1
    elif matchedQuery == 13:
        if(contextObj.visType != 0):
            taskType = contextObj.visType
        else:
            taskType = -1
        feature1 = parsedQuery[1]
        if(contextObj.fairFilter):
            subDf = df[df['id'].isin(contextObj.subsetId)]
            subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in contextObj.newId else 'unselected')
            textResponse = ['Suggested cutoff point is top '  + str(suggestFairness(feature1,subDf))]
        else:
            textResponse = ['Invalid Request. Fairness can only be calculated using filters involving the target.']
    elif matchedQuery == 14:
        taskType = 8
        textResponse = ['Showing the table']
    elif matchedQuery == 15:
     #free range response. Might be dangerous, buit we will see if it works
        textResponse = [genResponse(taskType, None, None, userInput,contextObj)[0]]
    elif matchedQuery == -9:
        textResponse = ['I am having some issues understanding you, can you please try again?']
    if(taskType != 0): #This is to update context object given a successful parse
        parsedInfo = []
        oldResponse = "" 
        if((matchedQuery in nonVisMatch) and (len(contextObj.responses)>0)):
            oldResponse = contextObj.responses[len(contextObj.responses)-1].old
            parsedInfo = contextObj.parsedInfo
        else:
            parsedInfo =  [feature1,processedInfo,num1,num2] #this is likely to cause some issues
            oldResponse = textResponse

        contextObj.newId = id
        contextObj.trackedVis = isTrackedVis
        contextObj.trackedName = trackedName
        contextObj.parsedInfo = parsedInfo
        contextObj.visType = taskType
        response = Response(id, oldResponse, textResponse, taskType, parsedInfo,userInput,staticResp)
        contextObj.responses.append(response)
        return contextObj
    else:
        return None
     

@app.callback(
    [Output("chat-output", "children"),
     Output("user-input", "value"),  # Reset user input
     Output('chat-history', 'data'),
     Output('features', 'data'),
     Output("submit-button", "n_clicks"),
     Output("keyboard", "n_keyups"),
     Output("chat-context", "data",allow_duplicate=True),

     ], 
    [Input("submit-button", "n_clicks"),
     Input("keyboard", "n_keyups"),
     ],
    [State("user-input", "value"),
     State('chat-history', 'data'),
     State("chat-context",'data')],
     running=[(Output("user-input", "disabled"), True, False),
              (Output("keyboard", "disabled"), True, False),
              (Output("submit-button", "disabled"), True, False),
              (Output("submitIcon", "src"), loadingIcon, arrowIcon)],
              
    prevent_initial_call=True
)

def update_chat(n_clicks, n_keyups, user_input, chat_history, chatContext):
    features = []
    if (n_clicks > 0 or n_keyups > 0) and user_input:
        thawedChatContext = jsonpickle.decode(chatContext)
        chat_history.append({'sender': 'user', 'message': [user_input]})
        currContext = processQuery(user_input,thawedChatContext)

        if(currContext == None):
            chat_history.append({'sender': 'computer', 'message': ['Error or invalid input, please try again']})
        else:
            features = currContext.features
            chat_history.append({'sender': 'computer', 'message': currContext.responses[len(currContext.responses)-1].new})
        user_input = ""
        n_clicks = 0
        n_keyups = 0
        frozenContext = jsonpickle.encode(currContext)

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
        
    return chat_output, user_input, chat_history,features,n_clicks, n_keyups, frozenContext

@app.callback(
    [Output("dummy-output", "children"),
     Output('chat-context','data',allow_duplicate=True)],  # Dummy output
    [Input("plot", "selectedData")],
    State('chat-context','data')
    ,
    prevent_initial_call=True
)
def display_selected_data(selected_data,chatContext):
    if selected_data:
        chatContext = jsonpickle.decode(chatContext)
        points = selected_data["points"]
        if points is not None and len(points) > 0:
            pointDf = pd.json_normalize(points)
            idList = pointDf['customdata'].tolist()
            flatIdList = [item for sublist in idList for item in sublist]
            chatContext.newId = flatIdList
            if(chatContext.visType == 4):
                chatContext.fairFilter = True
            else:
                chatContext.fairFilter = False
            chatContext = jsonpickle.encode(chatContext)
            return flatIdList,chatContext
    return no_update,no_update
    #else:
        #return "No points selected."


@app.callback(
    Output('plot', 'figure'),
    [Input('chat-history', 'data'),
     Input("dummy-output", "children"),
     Input('features', 'data')],
     State('chat-context','data')
)

def update_chart(chat_history,flatIdList, features,chatContext):
    #Instead of using df here, we will probably be better of pre-subsetting it
    if(chatContext != None and chatContext != 'null' ):
        chatContext = jsonpickle.decode(chatContext)
    else:
        return blank_fig()
    if(len(chatContext.responses) > 0 ):
        fig = blank_fig()
        if(chatContext.visType == -1):
            return blank_fig()
        subDf = df[df['id'].isin(chatContext.subsetId)]#subset df
        filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
        unfilt_df = prepareDF(subDf, 'unselected',features)
        df_melted = prepareDF(filtered_df, 'selected',features)

        result = pd.concat([df_melted,unfilt_df])
        
        if(chatContext.visType == 2):
            df_melted['rank'] = df_melted.groupby('feature')['y'].rank(ascending=False, method='first')
            tickVal = [int(df_melted['rank'].min()), int(df_melted['rank'].median()), int(df_melted['rank'].max())]
            reverse_labelalias = dict([(k,str(v)) for k,v in zip(tickVal,sorted(tickVal, reverse=True))])
            unfilt_df = unfilt_df[~unfilt_df['id'].isin(chatContext.newId)]
            if(not chatContext.trackedVis):
                legendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature", facet_col_wrap=2,custom_data=['id'],  category_orders={
                    "feature": [feature + "_v" for feature in features]},labels={"y" : chatContext.scoreName},)
                legendFig.update_traces(
                    hoverinfo='skip',
                    opacity=0,
                    marker=dict(
                                    color=df_melted["rank"],
                                    showscale=True,
                                    reversescale=True,
                                    colorbar=dict(
                                    title="Rank",
                                    tickmode="array",
                                    tickvals = tickVal,
                                    labelalias = reverse_labelalias,
                                    tickfont_size=20,
                                    title_font_size= 20,



                                )
                                ))
                fig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature", facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={
                    "feature": [feature + "_v" for feature in features]},labels={"y" : chatContext.scoreName})
                fig.update_traces(name='Rank',
                            marker=dict(
                                    color=df_melted["rank"],
                                    reversescale=False,
                                    showscale=False,
                                        colorbar=dict(
                                        title="Rank",
                                        tickmode="array",
                                    )
                                )
                            )
                fig = legendFig.add_traces(fig.data)

                unFig = px.scatter(unfilt_df, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={ 
                    "feature": [feature + "_v" for feature in features], "Selection": ["selected","unselected"]},labels={"y" : chatContext.scoreName})
            
                fig.update_yaxes(matches=None)
                trendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                    "feature": [feature + "_v" for feature in features]},labels={"y" : chatContext.scoreName})  
                
            else:
                tracked = chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]]
                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                oldFilter = prepareDF(oldFilter,'selected',features)
                featuresUsed = tracked.oldParse[0]
                legendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature", facet_col_wrap=2,custom_data=['id'],  category_orders={
                    "feature": [feature + "_v" for feature in features]})
                legendFig.update_traces(
                    hoverinfo='skip',
                    opacity=0,
                    marker=dict(
                                    color=df_melted["rank"],
                                    showscale=True,
                                    colorbar=dict(
                                    title="Rank",
                                    tickmode="array",
                                    tickvals = tickVal,
                            labelalias = reverse_labelalias,
                                                                tickfont_size=20,
                                    title_font_size= 20,

                                )
                                ))
                fig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature", facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={
                    "feature": [feature + "_v" for feature in features]})
                fig.update_traces(name='Rank',
                            marker=dict(
                                    color=df_melted["rank"],
                                    reversescale=False,
                                    showscale=False,
                                        colorbar=dict(
                                        title="Rank",
                                        tickmode="array",
                                    )
                                )
                            )
                fig = legendFig.add_traces(fig.data)

                unFig = px.scatter(unfilt_df, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], facet_col_wrap=2,custom_data=['id'], hover_data={'id': True},  category_orders={ 
                    "feature": [feature + "_v" for feature in features], "Selection": ["selected","unselected"]})
            
                fig.update_yaxes(matches=None)
                trendFig = px.scatter(df_melted, x="value", facet_col_spacing=0.04, y="y", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                    "feature": [feature + "_v" for feature in features]})
                trackTrend = px.scatter(oldFilter, x="value", facet_col_spacing=0.04, y="y", trendline_color_override="red", facet_col="feature",  facet_col_wrap=2, trendline="ols", category_orders={ 
                "feature": [feature + "_v" for feature in featuresUsed]})     
                trackTrend.update_traces(visible=False, selector=dict(mode="markers"))
                fig.add_traces(trackTrend.data)
            trendFig.update_traces(visible=False, selector=dict(mode="markers"))

            trendFig.update_xaxes(range=[0, None])
            trendFig.update_yaxes(range=[0, None])
            trendFig.update_traces(visible=False, selector=dict(mode="markers"))
            fig.add_traces(trendFig.data)
            fig.add_traces(unFig.data)
            fig.update_traces(selected=dict(marker=dict(color='#4590ff')), unselected=dict(marker=dict(color='rgba(44,69,107, 0.2)')))
            
        elif(chatContext.visType == 3):
            binEdge = getBins(unfilt_df,features)
            if(not chatContext.trackedVis):
                unfiltHis = processHistogram(unfilt_df, features, binEdge)
                newHis = processHistogram(df_melted, features, binEdge)
                newHis['Selection'] = 'Selected'
                unfiltHis['Selection'] = 'Unselected'
                newResult = pd.concat([unfiltHis,newHis])
                fig = px.bar(newResult, x="bins", y = "counts", facet_col="feature", facet_col_spacing=0.04, color="Selection", facet_col_wrap=2,  barmode="overlay", category_orders={ 
                  "feature": [feature for feature in features]}, color_discrete_sequence=['rgba(44,69,107, 0.2)','#0096FF'], )
                fig.update_layout(modebar_remove=['select', 'lasso'])

            else:

                #this one is really weird, for some reason plotly does not have smooth animations for histograms
                #so I am pre-processing the data to be a barchart pretending to be a histogram

                tracked = chatContext.responses[chatContext.trackedResponses[chatContext.trackedName]]
                oldFilter = subDf[subDf['id'].isin(tracked.oldId)]
                featuresUsed =tracked.oldParse[0]
                oldMelted = prepareDF(oldFilter, 'selected',features)
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
                    "feature": [feature  for feature in featuresUsed]},  animation_frame="oldNew", animation_group="bins")
                
                for i in range(len(featuresUsed)):
                    x = math.ceil(2/(i+(1-i%2)))
                    y = i%2+1
                    fig.update_yaxes(range= [0,graphRanges[i]], row=x, col = y)

              #  fig.update_yaxes(range= [0,graphRanges[0]], row=2, col = 1)
              #  fig.update_yaxes(range= [0,graphRanges[1]], row=2, col = 2)
               # fig.update_yaxes(range= [0,graphRanges[2]], row=1, col = 1) #Bottom Left
               # fig.update_yaxes(range= [0,graphRanges[3]], row=1, col = 2) #Bottom Right
        
                fig.update_layout(bargap=0.01)
                fig.update_layout(modebar_remove=['select', 'lasso'])

           



        elif(chatContext.visType == 4):
            rankDf = subDf.copy() 


            rankDf['Selection'] = 'unselected'
            rankDf.loc[rankDf['id'].isin(chatContext.newId), 'Selection'] = 'selected'
            rankDf['rank'] = rankDf['y'].rank(ascending=False, method='first')

            filDf = rankDf[rankDf['Selection'] == 'selected']
            unDf = rankDf[rankDf['Selection'] == 'unselected']
            filDf['colRank'] = filDf['y'].rank(ascending=False, method='first')
            legendFig = px.scatter(filDf, x="rank",  y="y",custom_data=['id'], hover_data={'id': True})
            legendFig.update_traces(
                    hoverinfo='skip',
                    opacity=0,
                    marker=dict(
                                    color=filDf["colRank"],
                                    showscale=True,
                                    colorbar=dict(
                                    title="Rank",
                                    tickmode="array",
                                    tickvals = [int(filDf["colRank"].min()), int(filDf["colRank"].max())],
                            labelalias = {int(filDf["colRank"].min()): 'Bottom', int(filDf["colRank"].max()) : 'Top'},
                                                                tickfont_size=20,
                                    title_font_size= 20,

                                )
                                ))
            fig = px.scatter(filDf, x="rank",  y="y",custom_data=['id'], hover_data={'id': True})
            fig.update_traces(name='Rank',
                            marker=dict(
                                    color=filDf["colRank"],
                                    reversescale=True,
                                    showscale=False,
                                        colorbar=dict(
                                        title="Rank",
                                        tickmode="array",
                                    )
                                )
                            )

            unFig = px.scatter(unDf, x="rank",  y="y", color='Selection', color_discrete_sequence=['#4590ff', 'rgba(44,69,107, 0.2)'], hover_data={'id': True},  category_orders={ 
                "Selection": ["selected","unselected"]})            
            fig = legendFig.add_traces(fig.data)

            fig.add_traces(unFig.data)

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
        elif(chatContext.visType == 8):
            rankDf = subDf.copy() 
            rankDf['Selection'] = 'unselected'
            rankDf.loc[rankDf['id'].isin(chatContext.newId), 'Selection'] = 'selected'
            rankDf = rankDf[rankDf.Selection != 'unselected']
            rankDf['rank'] = rankDf['y'].rank(ascending=False, method='first')
            rankDf = rankDf.drop(columns=['Selection','id'])
            rankDf = rankDf.loc[:, ~rankDf.columns.str.endswith('_a')]
            rankDf.columns = rankDf.columns.str.replace('_v$', '', regex=True)
            rankDf.rename(columns={'y': chatContext.scoreName}, inplace=True)
            rankDf = rankDf.sort_values(by = 'rank') 
            rankDf = rankDf[['rank'] + [x for x in rankDf.columns if x != 'rank']]
            fig = go.Figure(data=[go.Table(
            header=dict(values=list(rankDf.columns),
                        fill_color='#003459',
                        font=dict(color='white'),
                        align='left'),
            cells=dict(values=rankDf.transpose().values.tolist(),
               fill_color='#d3d3d3',
               align='left'))
            ])
        if(chatContext.visType != 8):
            fig.update_layout(showlegend=False)
        # fig.update_xaxes(range=[0, None])
        # fig.update_yaxes(range=[0, None])S
            fig.for_each_annotation(lambda a: a.update(font_size=17))
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_yaxes(showticklabels=True, row=1)
            fig.update_yaxes(showticklabels=True, row=2)
            fig.update_xaxes(showticklabels=True, row=1)
            fig.update_xaxes(showticklabels=True, row=2)

            fig.update_xaxes(showticklabels=True, row=1)
            fig.update_xaxes(showticklabels=True, row=2)
            fig.update_yaxes(title_font_size=18) 
            fig.update_xaxes(title_font_size=18) 
            fig.update_yaxes(matches=None) 
            fig.update_xaxes(matches=None)

        return fig
    else:
        return blank_fig()

if __name__ == "__main__":

    app.run_server(debug=False)

