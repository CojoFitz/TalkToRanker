import dash, math, warnings
from dash import dcc, html, callback, ctx
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
from dash_extensions import Keyboard
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
figureAb = 'assets/figureab.png'



adm = 'admission_all.csv'
credit = 'credit_risk_all.csv'
df = pd.read_csv(r'datasets/'+adm)
target = 'y'
chatContext = ContextObject()
nonVisMatch = (1,6,8,9,10,11,13)
chatContext.visType = 0
defaultIds = df['id'].to_list()
chatContext.subsetId = defaultIds
chatContext.newId =defaultIds
filtered_df = df[df['id'].isin(chatContext.newId)]
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

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_layout(modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'resetScale', 'autoScale', 'toImage'])
    return fig

def makeButton(feat, num):
    featureButton = dbc.Button(
        [
            feat,
            dbc.Badge("x", text_color="light"),
        ],
        color="primary",
        style={"width": "100px","font-size": "10px", "margin-top" : "45px"},
        id ={'role': feat, 'index': num}
    )
    return featureButton



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
                html.Th("Contextual Use Predictor Used", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
                html.Th("Contextual Use Predictor Not Used", style={'border': '1px solid black', 'width': size, 'textAlign': 'center'}),
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

introText = [""" Algorithmic rankers prove to be very useful in a multitude of different areas, as they assist greatly in the processes of making decisions. While those with data literacy may view ranker models as intuitive and easy to analyze, there is undeniably some level of knowledge required for one to obtain the answers to the questions they may have. As such, the average person may find it hard to utilize tools relating to rankings to the fullest extent. Our interface aims to make the process of utilizing rankers more accessible and intuitive. This is accomplished by leveraging, explainable artificial intelligence and visualizations to allow for rankers to be more accessible. """]
introHeader = 'Abstract'




overviewHeader = 'General Overview'
overviewBody = [
    """Our interface is a hybrid conversational and visualization interface,  which generates visual and textual representations relevant to questions about the dataset. The interface is split into two views, the chat view (a) and visualization view (b). """,
    html.Img(src=figureAb),
    """As you can see in the chat view, the user is able to ask questions about a dataset pertaining to university admissions. After the user asks a question two things occur, a textual response is generated in the chat view as well as a visual response in the visualization view. The exact process of how this occurs will be elaborated upon further in other sections. """,
    """Our interface is split into four major components being that of the Parser, Context Provider, Textual Explainer, and Visual Explainer. These components each play critical roles in shaping the interface's capabilities of delivering cohesive and informative information to the user. The following diagram highlights a overview of how they all interact with each other: """,
    html.Img(src=diagramImage)
]

parserHeader = 'Parser'
parserText = [
"""The interactions in this interface primarily result from the user’s input in the chat-view. Since there are many different ways a user can ask the same question, it is important for the interface to be able to distinctly understand a question. As such, there needs to be a way to determine what the question is asking and what information is needed to answer it. It is important to have such information to accurately generate textual and visual explanations for the user. To help accomplish this, we are utilizing large language models such as GPT 3.5 [1] via API calls from OpenAi, to assist in this process. The way how we utilize the large language models will be elaborated more upon, however, it is first important to consider related works. """,
"""One usage of LLMs in visualization interfaces is to generate executable code for visualizations. In an interface called Chat2Vis Paula and Susnjak demonstrated the capability of various LLMs to generate executable python code for the matplotlib library to create visualizations based on textual input [2]. This method was not selected to be used in our interface as there were some issues found with it. The code generated by the LLM was not always reliable, through testing this technique there were often times where the code would not execute or displayed wrong information. Additionally, with prompt engineering techniques that allow for instructions to be bypassed [3], a user can use this to inject whatever code they want. Furthermore, the unreliability and static nature of the visualizations makes the implementation of interactivity with the visualizations difficult to implement within reason.""",
"""We took inspiration from similar works [4,5] creating a question bank containing formatted questions known as a gold parse. The gold parse is an idealized format for different question types to be in, which in turn user inputs can be matched to. The following is an example of some of the gold parses in our question bank:""",
tableTest(),
"""The interface works by checking if the user’s input matches to the gold parse, if it does not it will use an LLM such as GPT 3.5 [1] to format and match the user’s input to match a gold parse. For example if a user types “I am very curious to know the correlation between the target and age”, the LLM will re-format their input as "What is the correlation of age with target". After the input is made to be in the format of the gold parse, it will then be parsed using a list of regular expressions for different query types. Each query type has different parameters, such as features and numbers, of which will be stored and the type of query will be stored as a numerical value. At this point, the user’s input is matched with a golden parse and all of the information necessary to generate a response and visualization has been parsed. The information obtained from the parser is then subsequently passed off to the context provider."""
]

contextHeader = 'Context Provider'
contextBody = ['Our interface produces interactions and explanations from both a textual and visual medium. These two components require a lot of contextual information about user requests, the dataset, and more in order to properly work. This interface addresses this with the use of the context provider, which produces and stores the necessary information for tasks to be carried out. ',
               """The context provider will receive data when a user makes an input in the chat view and their input is parsed by the parser. The parser sends the context provider information such as the task to be executed, features needed, any numerical values needed, and more. Once the context provider is given this information, most of it will be stored until it is needed by the explainers. The most actively used and changing aspect of the context provider is the global id list it has. Every item in a dataset used for the interface is expected to have a unique numeric id, the ids of selected points will be stored in the context provider based on interactions from the visualization view. The process of how selections work will be explained in more detail later in the section about the Visual Explainer, however, the context provider is what is responsible for maintaining a working memory of what the state is. 
 """,
 """The context provider also has a sub-component known as the contextual predictor, which serves to make predictions that assist in providing contextual information. The way how the subcomponents work will be elaborated more upon in sections more relevant to their usage, but in essence this sub-component makes use of LLMs to create string values that help with providing context. The need and type of context being used is unique for each aspect of the contextual predictor, which include: the contextualUse, contextualScore, and contextualElement. This area of the context provider may seem vague at first, but will be clarified when their uses are brought up later."""]


textualHeader = 'Textual Explainer'

textualInteraction = [f"""User:  “Hey, I am an admission counselor and I want to know if the correlation of the TOEFL_Score is important or not”""",
             f"""System: “Yes, the correlation between the target (likelihood of admission) and TOEFL Score is quite strong at 0.826, indicating that TOEFL Score is an important factor in predicting a student's likelihood of admission to the graduate program.”"""]

exampleFeats = [f""" "age", "gender", "height", "weight", "blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate", "respiratory_rate", "temperature", "cholesterol_total" """]

textualBody = ["""Our interface uses textual explanations generated with the assistance of LLMs such as GPT 3.5 [1] to provide rich textual responses. These textual responses are made using the textual explainer, whose function is as the name implies to provide explanations through text. The text explainer begins by receiving contextual information, such as filters, contextual use, and parsed information such as the task type and features. Once the textual explainer has the information it needs, it will generate what is called a static response based on the task type. The static response is essentially a template to generate a pre-made message that answers the question at its most basic level. The following table will show examples of the static response being generated based on some task types:""",
               explainerTable(),
               """The generation of the static response is important for the next step, of generating text using a large language model. Since a large language model is being used to generate text, an obvious question is why even bother making a static response when one can directly send the user’s input to the large language model. The static response exists to remedy the limitations large language models have with handling mathematical calculations and large amounts of data. By having the static response, we can provide the LLM with a baseline of what an accurate response looks like, and allow the LLM to build off of it. These textual responses are generated using GPT 3.5 [1] via the OpenAI api, by giving the LLM a query using both the user’s original unparsed message alongside the static response. The query will also include something known as the contextual use, but this will be explained in further detail later. The following is a complete example of an interaction between the user and the textual explainer:""",
               quoteBox(textualInteraction),
               """As can be observed in the interaction, the response was able to provide a mathematically based answer alongside an explanation for it. The user, who may not be as familiar with data science, can now know what the correlation is and why that may be important for their purposes. """,
               """As mentioned previously, the context provider also generates a string called the contextualUse, which is critical in making textual explanations more informed. The contextual use is generated in the context provider by giving an LLM a list of all the features in a dataset, and then asking the LLM to generate a short description of what the dataset is for. To demonstrate the effectiveness and need for this, let us assume we have a dataset with the following feature names:""",
                quoteBox(exampleFeats),
                """By looking at the features, it is quite easy to recognize that this dataset is based on medical information. The inclusion of features such as "blood_pressure_diastolic" and "heart_rate" help to distinguish the purpose of this dataset, as they are metrics that are commonly measured in medical environments. This dataset in particular highlights the need for the contextualUse, as the context of features such as “age” and “gender” can vary greatly depending on the dataset. To help highlight the need for the contextual use, we included an example of an identical question being asked with and without the contextual use predictor.""",
                html.Div([ boldText('Question:','What is the correlation of the target with age?'),
                boldText('Static Response:','The correlation between the target and age is 0.826'),
                boldText('Contextual Use (Generated via GPT 3.5[1]):', 'This dataset is likely used for monitoring and analyzing the health status of individuals, including assessing cardiovascular risk factors and overall well-being.'),
                                contextTable()
                ]),
                """As evident in the two responses generated above, responses generated by the contextual use predictor offer much more informative and relevant responses to the dataset as a whole. The contextual use predictor also allows for responses to be much more friendly towards users. Since a major goal is to make this interface accessible for individuals who may lack data literacy, the contextual use can enable responses to be clear and contextually aware. """
                ]
visualHeader = 'Visual Explainer'
visContextUse = ["""This dataset is likely used to predict or analyze graduate school admissions decisions based on applicants' academic scores, university ranking, statement of purpose, letters of recommendation, GPA, and research experience."""]

visualBody = [f"""The visual explainer's job is to offer a visual explanation for the questions that the users have about the ranker and data. In order to generate these visualizations we chose to use dash and plotly for our interface, as they provide tools for selection and animation. Visualizations are shown whenever a user asks a question that requires a visualization, visualizations being updated or not based on a user’s query is dependent on the task type. Similar to how textual responses in the textual explainer have a static response generated, the visual explainer will use a graph type for each specific type of question. For example, a correlation question will be met with a scatterplot and a distribution question will be met with a histogram. These visualizations also rely on the context provider for information such as the features, points, and selections needed to be shown on the visualization.""",
              f"""Since the interface is designed for rankers, visualizations that support scatter plots will give information about the ranks of points. The information is given based on coloring, wherein the higher ranked points are given darker colors and the lower ranked points are given lighter colors. This can be shown in figure c.): """,
                  html.Img(src=figureC),
                f"""Selections are another area of interactivity supported by our interface, and they allow for users to have more control over the data. Selections can be done by using a lasso and box select tool on the points of the graph you wish to see. Alternatively, a user can also request to perform a selection through the chat, the options for this will be elaborated upon in more detail later in this section. There are two types of selections that can be done as well, filtering and subsetting. Filtering will highlight the points that are selected and influence textual responses, however, the unselected points will remain present on the visualization with a grayed out color. Subsetting on the other hand allows for points to entirely be removed from the visualization as a whole, subsetting will also impact textual responses based on the group subsetted. These two functionalities are useful, as it allows users to cut out information not relevant or keep track of information for specific groups.""",
                f"""Since selecting with a lasso select tool isn’t always the most precise, textual selections prove to be useful for cases where a user may want a specific group selected. Here are the supported textual selection types in our interface, assume that all of these selections will be used for filtering:""",
                selectTable(),
                f"""As discussed in the context provider, every entry in the dataset is expected to have a unique numerical ID assigned to it. Since the context provider maintains a global record of the IDs, this allows for selections made on one visualization to persist into other visualizations, even if the type of visualization has changed. That means a user can perform a selection on a graph that shows the ranker's stability, and see how the selection they made impacted the distribution.This also applies for visualizations where there are more than one feature in the visualization, if one subplot in a view is selected, then the selections made will be reflect on the other plots.""",
                f"""As users begin to update filters a lot, they may also be interested in knowing how their filters may change things. A feature that we implemented was the ability to track responses. For example, let’s say a user wishes to keep track of the current state of selections for a scatter plot graph that shows correlation. They can request in the chat to “track the response as trackedName”, this will in turn keep track of what the previous response was alongside the visualization state of the response. As the user updates their selections, they can eventually ask to see the tracked response. Upon requesting the tracked response, they will be shown a visualization comparing the old from the new, alongside a chat message comparing the old and new response generated.""",
                f"""This visual explainer makes use of predictions made in the context provider, this is done to make the interface more descriptive. Since our interface expects datasets used with it to have a ‘y’ column, indicating the score, there is not always a clear indication of what y means. A solution of this is in the context provider through the contextScore. The contextScore is found by providing the contextualUse, as described in the textual explainer, to an LLM. From there the LLM will be asked to generate a descriptive name to replace ‘y’. This process is done prior to any query being executed and only executed once, so as such a user’s input will not influence the name of y in any capacity. Let us see an example where a visualization was generated showing the correlation between the TOEFL Score and the target(y):""",
                  html.Img(src=figureC),
                html.Div([
                    html.P('As seen above, using the contextual predictor, the LLM was able to identify that the Y score likely refers to the admission decisions with this dataset. The contextualUse here was:'),
                    quoteBox(visContextUse),
                    html.P("""The contextual use in this scenario helped to identify a suitable name for the value of y, that is informative and practical for the purposes of the user. """)
                    ])
            ]




    
app.layout = html.Div([


html.Div(
    children=[
        html.H1(
            "Introduction",
            style={
                'text-align': 'center', 'margin-top' : '40px'}
        ),
        html.Div(
            """ Algorithmic rankers prove to be very useful in a multitude of different areas, as they assist greatly in the processes of making decisions. While those with data literacy may view ranker models as intuitive and easy to analyze, there is undeniably some level of knowledge required for one to obtain the answers to the questions they may have. As such, the average person may find it hard to utilize tools relating to rankings to the fullest extent. Our interface aims to make the process of utilizing rankers more accessible and intuitive. This is accomplished by leveraging, explainable artificial intelligence and visualizations to allow for rankers to be more accessible. """,
            style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-bottom' : '90px',
                'margin-left' : '20%',
                'margin-right' : '20%',
                'font-size' : '20px'

            }
        )
    ,
    textSection(overviewHeader,overviewBody),
    textSection(parserHeader,parserText),
    textSection(contextHeader,contextBody),
    textSection(textualHeader, textualBody),
    textSection(visualHeader, visualBody),
    textSection('Sources',[sources()]),
    textSection('Try the interface out for yourself: ',''),
    textSection('',''),


    

],
),
html.Div(["Enter OpenAi Api Key:",
            html.Div(dcc.Input(id='apikey', type='password')),
            html.Button('Submit', id='apiButton'),
            html.Div(id='output-container-button',
                    children='Enter a value and press submit')
        ],style={"display": "flex", "align-items": "center", " flex-direction": "column-reverse","justify-content": "center"}),

    
    
    
    html.Div([
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
        dcc.Store(id='features', storage_type='local', data=[]),

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
    Output('output-container-button', 'children'),
    Input('apiButton', 'n_clicks'),
    State('apikey', 'value'))


def update_output(n_clicks, value):
    if(value is not None):
        apiKey = value

        predictedContext = contextualPredictor(allFeatures, apiKey)
        contextualUsage = predictedContext.predictContextualUse()
        scoreName = predictedContext.contextualizeScore(contextualUsage)
        elementName = predictedContext.contextualizeElement()

        chatContext.apiKey = apiKey
        chatContext.elementName = elementName
        chatContext.scoreName = scoreName
        chatContext.contextUse = contextualUsage

numMessages = 0




def genResponse(taskType, df, feature1,userInput):
    explainer = creativeExplainer(chatContext.contextUse, chatContext.apiKey)
    genResp = 'None'
    if(taskType == 0):
        genResp = 'Query not understood'
    elif(taskType == 2):
        genResp = []
        for feat in feature1:
            cor = round(df[feat+'_v'].corr(df[target]),3)
            genResp += ["The correlation between the target and " + feat + ' is ' + str(cor)+'.'] #Correlation
        
        staticInput = ' '.join(genResp)
        genResp = [explainer.explainGen(userInput,staticInput)]

    elif(taskType == 3):
        genResp = []
        for feat in feature1:
            std = round(df[feat+'_v'].std(),3)
            mean = round(df[feat+'_v'].mean(),3)
            median = round(df[feat+'_v'].median(),3)
            staticInput = ['Information about feature: ' + feat, 'Standard Deviation: ' + str(std), 'Mean: ' + str(mean), 'Median: ' + str(median)]
            staticInput = ' '.join(staticInput)
            genResp = [explainer.explainGen(userInput,staticInput)]

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
    showFeatures = feature
    #for feat in feature:
      #  if (feat in showFeatures):

 #   if(feature not in showFeatures):
    #    if(len(showFeatures)>=4):
      #      showFeatures.pop()
       # showFeatures.insert(0,feature)
      #  getButtons(showFeatures)\

def processQuery(userInput, contextObj):
    isTrackedVis = False
    trackedName = ''
    taskType = 0 #Zero will be our error state
    textResponse = []
    feature1 = processedInfo = num1 = num2 = None
    id = contextObj.newId
    #I will need to add the patterns for the other tasks
    formatQuery = matchResponse(allFeatures,chatContext.apiKey)
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


        textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ chatContext.elementName +' matched')
    elif matchedQuery == 2 or matchedQuery == 3:
        taskType = matchedQuery
      #Example: "What is the correlation of the target with GRE_Score_v"
      #"What is the correlation of the target with TOEFL_Score_v"
        feature1 = parsedQuery[1]
        if(set(feature1).issubset((set(featureDict['all'])))):
            contextObj.features = feature1
            subDf = df[df['id'].isin(chatContext.subsetId)]
            filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
           # cor = filtered_df[feature1+'_v'].corr(filtered_df[target])
           # textResponse = "The correlation between the target and " + feature1 + ' is ' + str(cor)+'.'
            textResponse += (genResponse(taskType, filtered_df, feature1, userInput))
        else:
            textResponse.append("One of your entered features has an error"  + ". Please check the spelling or verify that it is a feature.")
            feature1 = None
            taskType = -1


    elif matchedQuery == 4:
        taskType = 4
        textResponse.append(genResponse(taskType, None, None, userInput))
    elif matchedQuery == 5:
        taskType = 5 #Feature attribution stuff
        subDf = df[df['id'].isin(chatContext.subsetId)]
        filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
        textResponse.append(genResponse(taskType, filtered_df, None,userInput))
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
                contextObj.features = feature1
            subDf = df[df['id'].isin(chatContext.subsetId)]
            filtered_df = subDf[subDf['id'].isin(chatContext.newId)]
            textResponse = textResponse + ["Old response is: " ]+ oldResponse.old + ['New Response is:'] + genResponse(taskType, filtered_df,feature1,userInput)
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
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ chatContext.elementName +' matched')
        else:
            textResponse.append('Error, theere is no categorical feature: ' + col + ' please double check your query!')
    elif matchedQuery == 9:
        taskType = 6
        textResponse = genResponse(taskType, df, None,userInput)
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
        textResponse.append('Getting top ' + str(topNum) + ' ' + chatContext.elementName)
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
            textResponse.append('Applying ' + mode +', ' +  str(numEntries) +' '+ chatContext.elementName +' matched')
    elif matchedQuery == 12:
        feature1 = parsedQuery[1]
        taskType = 7
        if(contextObj.fairFilter):
            subDf = df[df['id'].isin(chatContext.subsetId)]
            subDf['selection'] = subDf['id'].apply(lambda x: 'selected' if x in chatContext.newId else 'unselected')
            textResponse = genResponse(taskType,subDf,feature1,userInput)
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
            parsedInfo =  [feature1,processedInfo,num1,num2] #this is likely to cause some issues
            oldResponse = textResponse

        contextObj.newId = id
        contextObj.trackedVis = isTrackedVis
        contextObj.trackedName = trackedName
        contextObj.parsedInfo = parsedInfo
        contextObj.visType = taskType
        response = Response(id, oldResponse, textResponse, taskType, parsedInfo,userInput)
        contextObj.responses.append(response)
        return contextObj
    else:
        return None

@app.callback(
    [Output("chat-output", "children"),
     Output("user-input", "value"),  # Reset user input
     Output('chat-history', 'data'),
     Output('features', 'data'),

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
    features = []
    if (n_clicks > 0 or n_keyups > 0) and user_input:
        # Append the user's message to the chat history
        # Empty user input
        chat_history.append({'sender': 'user', 'message': [user_input]})
        currContext = processQuery(user_input,chatContext)

        if(currContext == None):
            chat_history.append({'sender': 'computer', 'message': ['Error or invalid input, please try again']})
        else:
            features = currContext.features
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
        
    return chat_output, user_input, chat_history,features

@app.callback(
    Output("dummy-output", "children"),  # Dummy output
    [Input("plot", "selectedData")]
)
def display_selected_data(selected_data):
    if selected_data:
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
            return flatIdList
    else:
        return "No points selected."


@app.callback(
    Output('plot', 'figure'),
    [Input('chat-history', 'data'),
     Input("dummy-output", "children"),
     Input('features', 'data')],
)

def update_chart(chat_history,flatIdList, features):
    #Instead of using df here, we will probably be better of pre-subsetting it. 
    if(len(chatContext.responses) > 0 ):
            
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
