from flask import Flask, session
from flask_session import Session
from dash import Dash, dcc, html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
# to delete dash_table
from dash import html, dcc, dash_table
# import urllib.parse

import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
# import pinecone

#################################################
#####     configurations
#################################################

##### openai-mais2
API_KEY = "d70b34fbd24d4016a5cf88dbc5f91e78"
RESOURCE_ENDPOINT = "https://openai-mais-2.openai.azure.com/"
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-07-01-preview"

#################################################
#####     Load data 
#################################################

# load markdown file for About page
with open('data/about.md', 'r') as markdown_file:
    markdown_about = markdown_file.read()

# tags for topic keywords
tags = {
    'Africa':               'Africa trade, African Continental Free Trade Area (AfCFTA)',
    'COVID-19':             'covid, vaccine, diagnostics, therapeutics',
    'Digital trade':        'digital trade, ecommerce, moratorium on electronic transmissions',
    'E-commerce':           'ecommerce',
    'Environment':          'environment, climate change, polution, environmental protection, biodiversity',
    'Geopolitics':          'geopolitics, US-China, trade war, frictions, geopolitical',
    'Global economy':       'global economy, GDP growth, trend, outlook, forecast',
    'Intellectual property':'intellectual property rights, copyright',
    'MSME':                 'micro small and medium enterprises',
    'Subsidies':            'industrial subsidies grant',
}


# tags for topic keywords
styles = {
    'Delegates/Heads of states':    'speak to delegates/heads of states: Diplomatic, formal, strategic, respectful, authoritative, policy-oriented, persuasive, factual, concise, collaborative',
    'Think tanks':                  'speak to think tanks: Diplomatic, Strategic, Authoritative, Analytical, Persuasive, Forward-thinking, Inclusive, Policy-focused, Insightful, Collaborative',
    'Academics':                    'speak to academics: Scholarly, Analytical, Informed, Thought-provoking, Collaborative, Insightful, Respectful, Comprehensive, Evidence-based, Innovative',
    'Students':                     'speak to students: Inspirational, engaging, informative, motivational, relatable, empathetic, uplifting, visionary, accessible, encouraging',
}

speechdb = pd.read_parquet('data/speech-text-embedding.parquet')
contextdb = speechdb[speechdb['n_tokens']>50].copy()
speechlist = speechdb.groupby(['Subfolder','FileName']).size().reset_index(name='NParas')
speechlist.columns = ['Folder','File Name','Number of paragraphs']

#################################################
##### Speech app
#################################################

# search through the reviews for a specific product
def search_speech_db(df, user_query, ncontext=20):
    embedding = get_embedding(
        user_query,
        engine="test-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(ncontext)
    )
    return res

def generate_context(topic, ncontext=20):
    res = search_speech_db(contextdb, topic, ncontext=ncontext)
    return res['Text'].to_list(), res['similarities'].min(), res['similarities'].max()

def build_prompt_with_context(topic, context=[], nwords=300, audience='government officials', additional='None'):
    return [{'role': 'system', 
             'content': f'''As a speech writer, you are tasked with composing a speech for the Director General of the World Trade Organization. \
                            The speech should address the specific topic provided by the user, incorporating relevant contexts and information as mentioned. \
                            User may also provide addtional requirement, background information, or outlines.
                            Ensure that the speech used in the style suggested by user. A general rule is to be persuasive, informative, and use convincing figures. \
                            Additionally, the speech should be tailored to meet the exact length requirement set by the user, specified in the number of words. \
                            Your task is to write a speech that effectively conveys the WTO's perspective on the given topic, while maintaining the Director General's tone and style.'''
                    }, 
            {'role': 'user', 
             'content': f"""
                        Topics:
                        {topic} \
                            
                        Use the following contexts:
                        {' '.join(context)} \

                        Follow the additional instructions or outlines or use the additional information as provided below:
                        {additional} \

                        Adjust the contents and tone for targeted audience:
                        {audience} \

                        The number of words in the speech should be:
                        {nwords} words \

                        Speech:
            """}]

def write_speech(message, temperature=0, model="gpt-35-turbo-16k"):
    response = openai.ChatCompletion.create(
        engine=model,
        messages=message,
        temperature=temperature,
        max_tokens=3000,
    )
    # Strip any punctuation or whitespace from the response
    return response.choices[0].message.content.strip('., ')


#################################################
##### Dash App
#################################################

# Hardcoded users (for demo purposes)
USERS = {"admin": "admin", "ersd": "ersd", "w": "w"}

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecretkey'
server.config['SESSION_TYPE'] = 'filesystem'

Session(server)

# dash app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = Dash(__name__, server=server, 
           external_stylesheets = external_stylesheets,
           suppress_callback_exceptions=True
           )

app.title = 'Speech Database'
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

### sidebar
sidebar_header = dbc.Row([
    html.A([dbc.Col(html.Img(src=app.get_asset_url("logo.png"),  width="180px", style={'margin-left':'15px', 'margin-bottom':'50px'}))], href="/page-1"),
    dbc.Col(
        html.Button(
            # use the Bootstrap navbar-toggler classes to style the toggle
            html.Span(className="navbar-toggler-icon"),
            className="navbar-toggler",
            # the navbar-toggler classes don't set color, so we do it here
            style={
                "color": "rgba(0,0,0,.5)",
                "bordercolor": "rgba(0,0,0,.1)",
            },
            id="toggle",
        ),
        # the column containing the toggle will be only as wide as the
        # toggle, resulting in the toggle being right aligned
        width="auto",
        # vertically align the toggle in the center
        align="center",
    ),
])

sidebar = html.Div([
                    sidebar_header,
                    # use the Collapse component to animate hiding / revealing links
                    dbc.Collapse(
                        dbc.Nav([
                                dbc.NavLink("Write ", href="/page-1", id="page-1-link"),
                                dbc.NavLink("Search", href="/page-2", id="page-2-link"),
                                dbc.NavLink("Browse by topics", href="/page-3", id="page-3-link"),
                                dbc.NavLink("Speech List", href="/page-4", id="page-4-link"),
                                dbc.NavLink("About", href="/page-5", id="page-5-link"),
                                dbc.NavLink("Logout", href="/logout", active="exact"),  # Add a logout link
                            ], vertical=True, pills=False,
                        ), id="collapse",
                    ),
                    html.Div([html.P("V0.2 (20240131)",
                                # className="lead",
                            ),],id="blurb-bottom",
                    ),
                ], id="sidebar",
            )

content = html.Div(id="page-content")

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 6)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 6)]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Location(id='logout-url', refresh=False),  # Added logout URL component
    # login facet
    dbc.Container(
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Sign in to Speech Database", className="card-title"),
                            html.Br(),
                            dbc.Form(
                                [
                                    dbc.Row([
                                            dbc.Col([
                                                    dbc.Input(type="text", id="username", placeholder="Username", style={"width": 300}),
                                                ], className="mb-3",
                                            )
                                        ]
                                    ),
                                    dbc.Row([
                                            dbc.Col([
                                                    dbc.Input(type="password",  id="password", placeholder="Password",style={"width": 300}),
                                                ], className="mb-3",
                                            )
                                        ]
                                    ),
                                    dbc.Button(id='login-button', children='Sign in', n_clicks=0, color="primary", className="my-custom-button", style={"width": 300}),
                                ], 
                            ),
                        ], className="d-grid gap-2 col-8 mx-auto",
                    ),
                    className="text-center",
                    style={"width": "500px", "margin": "auto", "background-color": "#e4f5f2"},
                ), width=6, className="mt-5",
            )
        ), id='login-facet',className="login-page",
    ),
    html.Div([sidebar, content], id='page-layout', style={'display': 'none'}),
])

@app.callback(
    [Output('login-facet', 'style'),
     Output('page-layout', 'style')],
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value')]
)
def update_output(n_clicks, username, password):
    if n_clicks > 0:
        if username in USERS and USERS[username] == password:
            session['authed'] = True
    if session.get('authed', False):
        return  {'display': 'none'}, {'display': 'block'}
    else:
        return {}, {'display': 'none'}

# render content according to path
@app.callback(Output("page-content", "children"),
              Output("logout-url", "pathname"),  # Added callback output for logout URL
              [Input("url", "pathname"), Input("logout-url", "pathname")])
def render_page_content(pathname, logout_pathname):
    if logout_pathname == "/logout":  # Handle logout
        session.pop('authed', None)
        session.clear()
        return dcc.Location(pathname="/login", id="redirect-to-login"), "/logout"

    elif pathname in ["/","/login", "/page-1"]:
        return html.Div([
            html.H4("Draft a speech based on the requirement and previous speeches as contexts", ),
            html.Br(),
            html.H6("Topics, keywords:"),
            dbc.Row([
                dbc.Col(
                    dbc.InputGroup([
                            dbc.Input(id="write-input-box", type="text", placeholder="Enter a topic: e.g. globalization OR digital trade"),
                    ])
                )], justify="center", className="header", id='search-container1',
            ),

            html.Br(),
            html.H6("Additional (requirements, information, contexts, outlines): "),
            dbc.Row([
                dbc.Col(
                    dbc.Textarea(id="write-textarea-additional",  placeholder="Enter a topic: e.g. globalization OR digital trade", size="md",rows=4, style={"width": "100%"})
                )
                ], justify="center", className="header", id='search-container2', 
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.H6('Model: '), width=2, style={'margin-top':5,'margin-left':0}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="write-radio-select-model",
                            options=[
                                {"label": 'ChatGPT 3.5 16k', "value": 'gpt-35-turbo-16k'},
                                {"label": 'ChatGPT 4', "value": 'gpt-4'},
                            ],
                            value='gpt-35-turbo-16k',
                            inline=True,
                        ),
                        width=4,
                    ),

                    dbc.Col(html.H6('Length (words):'), width=2, style={'margin-top':5,'margin-left':0}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="write-radio-select-words",
                            options=[
                                {"label": "300", "value": 300},
                                {"label": "500", "value": 500},
                                {"label": "1000", "value": 1000},
                                {"label": "1300", "value": 1300},
                            ],
                            value=500,
                            inline=True,
                        ),
                        width=4,
                    ),
                ],
                align="center",
                style={"margin-bottom": "0px"},
            ),


            dbc.Row(
                [
                    # dbc.Col(html.P("Degree of randomness (temperature): Increase for more creativity"), width="100px",  style={'margin-top':5,'margin-left':10}),

                    dbc.Col(html.H6(["Degree of randomness (temperature): ", html.Br(), 
                                    "Increase for more creativity"]), 
                                    width=6,  style={'margin-top':5,'margin-left':0}),

                    dbc.Col(
                        dcc.Slider(0, 1, 0.2, value=0.4, id='write-slider-temperature'),style={"margin-top": "20px"}, width=6,
                        # dcc.Slider(0, 1,
                        #             step=None,
                        #             marks={
                        #                 0: '0',
                        #                 0.25: '0.25',
                        #                 0.5: '0.5',
                        #                 0.75: '0.75',
                        #                 1: '1'
                        #             },
                        #             value=0.5
                        #         )
                    ),
                ],
                align="center",
                style={"margin-bottom": "0px"},
            ),


            dbc.Row(
                [
                    dbc.Col(html.H6('Number of paras as input: '), width=3, style={'margin-top':2,'margin-left':0}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="write-radio-select-context",
                            options=[
                                {"label": '20', "value": 20},
                                {"label": '30', "value": 30},
                                {"label": '50', "value": 50},
                            ],
                            value=30,
                            inline=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H6('Audience and style'),
                            dcc.Dropdown(
                                id='write-dropdown-style',
                                multi=False,
                                options=[{'label': i[0], 'value': i[1]} for i in styles.items()],
                                value='speak to delegates/heads of states: Diplomatic, formal, strategic, respectful, authoritative, policy-oriented, persuasive, factual, concise, collaborative',
                                clearable=False
                            ),
                        ]
                    )                    
                ],
                align="center",
                style={"margin-bottom": "10px"},
            ),

            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Draft speech ...", id="write-submit-button", n_clicks=0, color='info'), width={'size': 12}, className='text-right'
                        ),
                ],
                align="right",
                # style={"margin-bottom": "1px"},
            ),


            html.Hr(),

            # dbc.Row([
            #     dbc.Col([
            #         html.P("Sample topics: "),
            #     ])
            # ]),


            # dbc.Row([

            #     dbc.Col([
            #         html.P("Sample topics: "),
            #         dcc.Markdown('''
            #                     - Trade and environment
            #                     - Globalization and re-globalization
            #                     - WTO and multilateral trading system
            #                     - US and China trade war
            #                     '''
            #             ),
            #     ], width=6),
            #     dbc.Col([
            #         dcc.Markdown('''
            #                     - Industrial policy
            #                     - Subsidies
            #                     - Least developed country and trade
            #                     - Africa and trade
            #                     - Digital trade '''
            #             ),
            #     ], width=6),

            # ], justify="center", className="header", id='write-sample-topics'),


            dbc.Row([
                dbc.Row(
                    dbc.Col(
                        html.H6("Sample topic: "),
                        width=12
                    )
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Markdown('''
                            - Trade and environment
                            - Globalization and re-globalization
                            - WTO and multilateral trading system
                            - US and China trade war
                        '''),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Markdown('''
                            - Industrial policy
                            - Subsidies
                            - Least developed country and trade
                            - Africa and trade
                            - Digital trade
                        '''),
                        width=6
                    ),
                ]),
            ], justify="center", className="header", id='write-sample-topics'),














            html.Br(),
            dbc.Row([ 
                # html.Div(id="search-results", className="results"),
                dbc.Col([
                        dcc.Loading(id="loading2", type="default", children=html.Div(id="write-results"), fullscreen=False),
                    ], width=12),
            ], justify="center"),
        ]), pathname

    elif pathname == "/page-2":
        return dbc.Container([
            html.H6("Search SpeechDB with embeddings", className="display-about"),
            html.Br(),
            html.Br(),            
            dbc.Row([
                dbc.Col(
                        dbc.InputGroup([
                                dbc.Input(id="search-box", type="text", placeholder="Enter search query, e.g. subsidies, climate change"),
                                dbc.Button(" Search ", id="search-button", n_clicks=0,
                                                #    className="btn btn-primary mt-3", 
                                            ),
                            ]
                        ), width=12,
                    ),
                ], justify="center", 
                # className="header", 
                id='search-container'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.Label("Display paragraphs with the highest score:"), width="auto",  style={'margin-top':5,'margin-left':10}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="radio-select-top",
                            options=[
                                {"label": "Top 20", "value": 20},
                                {"label": "Top 50", "value": 50},
                                {"label": "Top 100", "value": 100},
                            ],
                            value=50,
                            inline=True,
                        ),
                        width=True,
                    ),
                ],
                align="center",
                style={"margin-bottom": "10px"},
            ),
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(
                        '''
                        Search query examples:
                        * subsidies and government supports
                        * MSME, SME or small businesses in Africa
                        * Trade finance
                        * Africa
                        '''
                        ),
                ], width=12),
            ], justify="center", 
            # className="header", 
            id='sample-queries'),

            html.Br(),
            html.Br(),

            dbc.Row([ 
                # html.Div(id="search-results", className="results"),
                dbc.Col([
                        # html.Div(id="search-results", className="results"),
                        dcc.Loading(id="loading", type="default", children=html.Div(id="search-results"), fullscreen=False),
                    ], width=12),
            ], justify="center"),
        ]), pathname
    
    elif pathname == "/page-3":
        return dbc.Container([
            html.H6("Browse reports by topics", className="display-about"),
            html.Br(),
            html.Div(id='tag-container', children=[dbc.Button(key, id={'type': 'tag', 'index': i}, color="light", className="me-1", style={'margin-right':'10px', 'margin-bottom':'10px'}) for i, key in enumerate(tags)]),
            html.Br(),
            dbc.Row([ 
                dbc.Col([
                        # html.Div(id="search-results", className="results"),
                        dcc.Loading(id="loading", type="default", children=html.Div(id="search-results3"), fullscreen=False),
                    ], width=12),
            ], justify="center"),
        ]), pathname

    elif pathname == "/page-4":
        return html.Div([
                html.H6("Speeches in the database", className="display-about"),
                html.P(''),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in speechlist.columns],
                    data=speechlist.to_dict('records'),
                    style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(240, 240, 240)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(210, 210, 210)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    }
                )
            ]), pathname

    elif pathname == "/page-5":
        return html.Div([
                            html.H4("About the tools and the Speech Database", className="display-about"),
                            html.Br(),
                            dcc.Markdown(markdown_about, id='topic',
                                         style={
                                            "display": "inline-block",
                                            "width": "100%",
                                            "margin-left": "0px",
                                            "align": "left",
                                            # "verticalAlign": "top"
                                        }),
                ]), pathname

    else:
        return html.P("404: Not found"), pathname











#################################################
#####     Page Write
#################################################

# call back for returning results
@app.callback(
        [Output("write-results", "children"),  
         Output("write-sample-topics", "style")
        ],
        [Input("write-submit-button", "n_clicks"),
        #  Input("write-input-box", "n_submit")
        ], 
        [State("write-input-box", "value"),
         State('write-radio-select-context', 'value'),
         State('write-radio-select-model', 'value'),
         State('write-radio-select-words', 'value'),
         State('write-slider-temperature', 'value'),
         State('write-dropdown-style', 'value'),
         State('write-textarea-additional', 'value'),
         ]
)
# def write_speech(n_clicks, n_submit, topic, ncontext, nwords, temperature):
def write_draft_speech(n_clicks, topic, ncontext, model, nwords, temperature, audience, additional):

    # Check if the search button was clicked

    # if (n_clicks <=0 and n_submit is None) or search_terms=='' or search_terms is None:
    # if (n_clicks <=0 and n_submit is None) or topic=='' or topic is None:
    if n_clicks  <=0  or n_clicks is None or topic=='' or topic is None:
        return "",  None
    else:

        # return html.Div(
        #             dbc.Container(
        #                 [
        #                     dbc.Row(
        #                         [html.P('Draft (' + str(nwords) + ' words): ' + 'topic = "' + topic + '" and Temperature = ' + str(temperature) )],
        #                         justify="between",
        #                         style={"margin-bottom": "5px"},
        #                     ),
        #                     # dbc.Row(
        #                     #     [html.P(dcc.Markdown(draft))],
        #                     #     justify="between",
        #                     # ),
        #                 ],
        #             )
        #         ),  {'display': 'none'}


        # ncontext = 20

        # audience = 'delegates to the WTO'
        # model="gpt-4"
        # topic = 'reglobalization'
        # topic = 'trade under most favoriate nation principle'
        # topic = 'ecommerce'
        # topic = 'trade in Africa'
        # topic = 'Trade and environment'
        # topic = "China and US trade war"
        # topic = topic
        # ncontext = 20
        context, c_min, c_max = generate_context(topic, ncontext)

        context1 = ' '.join(context)

        # nwords = 300
        message = build_prompt_with_context(topic, context, nwords, audience, additional)
        # print(message)
        # message1 = ' '.join(message)

        draft = 'empty draft'
        # temperature = 0
        draft = write_speech(message, temperature, model)

        print(str(len(context1.split())), str(len(draft.split())))
    return html.Div(
                dbc.Container(
                    [
                        dbc.Row(
                            [html.P('Draft (' + str(len(draft.split()))  +" words): " + 'topic = "' + topic + \
                                    '", temperature = ' + str(temperature) + ', context min score =' + str(c_min) +\
                                        ', target words =' + str(nwords) + ', medel =' + str(model)
                                        )],
                            justify="between",
                            style={"margin-bottom": "5px"},
                        ),
                        dbc.Row(
                            [html.P(dcc.Markdown(draft))],
                            justify="between",
                        ),
                    ],
                )
            ),  {'display': 'none'}












































#################################################
#####    Page Search
#################################################

# call back for returning results
@app.callback(
        [Output("search-results", "children"),  
        #  Output("top-space", "style"),
         Output("sample-queries", "style")
         ],
        [Input("search-button", "n_clicks"),
         Input('search-box', 'n_submit'), ], 
        [State("search-box", "value"),
        State('radio-select-top', 'value')]
        )
def search(n_clicks, n_submit, search_terms, top):
    # Check if the search button was clicked
    if (n_clicks <=0 and n_submit is None) or search_terms=='' or search_terms is None:
        return "",  None
    else:
        df = search_speech_db(speechdb, search_terms, ncontext=top)
        df['meta'] = df['FileName'] + '\n Para: ' + df['ParagraphID'].astype(str) + '\n Score: ' + df['similarities'].astype(str) 
        df['text'] = df['Text']

        matches = df[['meta', 'text']]
        matches.columns = ['Meta','Text (Paragraph)']

        # Display the results in a datatable
        return html.Div(style={'width': '100%'},
                        children=[
                            # html.P('topic: ' + str(len(matches)) +" paragraphs, with score ranging from " + str(df['score'].min()) + ' to ' + str(df['score'].max())),
                            # html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank",),
                            html.Br(),
                            dbc.Row(
                                [
                                    # dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['similarities'].min()) + ' to ' + str(df['similarities'].max())), width={"size": 9, "offset": 0}),
                                    # dbc.Col(html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank"), width={"size": 3, "offset": 0}),
                                ],
                                justify="between",
                                style={"margin-bottom": "20px"},
                            ),

                            html.Br(),
                            dash_table.DataTable(
                                    id="search-results-table",
                                    columns=[{"name": col, "id": col} for col in matches.columns],
                                    data=matches.to_dict("records"),

                                    editable=False,
                                    # filter_action="native",

                                    sort_action="native",
                                    sort_mode="multi",
                                    
                                    column_selectable=False,
                                    row_selectable=False,
                                    row_deletable=False,
                                    
                                    selected_columns=[],
                                    selected_rows=[],
                                    
                                    page_action="native",
                                    page_current= 0,
                                    page_size= 20,
                                    style_table={'width': '900px'},
                                    style_header={'fontWeight': 'bold'},
                                    style_cell={
                                        # 'height': 'auto',
                                        # 'minWidth': '50px', 
                                        # 'maxWidth': '800px',
                                        # # 'width': '100px',
                                        # 'whiteSpace': 'normal',
                                        'textAlign': 'left',
                                        'fontSize': '14px',
                                        'verticalAlign': 'top',
                                        'whiteSpace': 'pre-line'
                                    },
                                    style_cell_conditional=[
                                        # {'if': {'column_id': 'Symbol'},
                                        #  'width': '50px'},
                                        # {'if': {'column_id': 'Member'},
                                        #  'width': '90px'},
                                        # {'if': {'column_id': 'Date'},
                                        #  'width': '80px'},
                                        # {'if': {'column_id': 'Section/Topic'},
                                        #  'width': '200px'},
                                        {'if': {'column_id': 'Text (Paragraph)'},
                                        'width': '1000px'},
                                        # {'if': {'column_id': 'Score'},
                                        #  'width': '80px', 'textAlign': 'right'},
                                    ],
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(250, 250, 250)',
                                        }
                                    ],
                                    style_as_list_view=True,
                                )
                            ]
                ),  {'display': 'none'}

#################################################
#####     Browse by Topic
#################################################

@app.callback(
    # Output('table', 'data'),
    Output("search-results3", "children"),
    [Input({'type': 'tag', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State({'type': 'tag', 'index': dash.dependencies.ALL}, 'children')]
)
def update_table(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return None # df.to_dict('records')

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    tag_clicked = ctx.states[button_id + '.children']

    df = search_speech_db(speechdb, tags[tag_clicked], ncontext=50)
    # df['meta'] = df['FileName'] + '\n' + df['symbol'] + '\n' + df['date'] + '\n Score: ' + df['score'].astype(str) 
    df['meta'] = df['FileName'] + '\n Para: ' + df['ParagraphID'].astype(str) + '\n Score: ' + df['similarities'].astype(str) 
    df['text'] = df['Text']

    matches = df[['meta', 'text']]
    matches.columns = ['Meta','Text (Paragraph)']


    # Display the results in a datatable
    return html.Div(style={'width': '100%'},
                    children=[
                        # html.P('Find ' + str(len(matches)) +" paragraphs, with score ranging from " + str(df['score'].min()) + ' to ' + str(df['score'].max())),
                        # html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank",),
                        html.Br(),
                        dbc.Row(
                            [
                                # dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['similarities'].min()) + ' to ' + str(df['similarities'].max())), width={"size": 9, "offset": 0}),
                                # dbc.Col(html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank"), width={"size": 3, "offset": 0}),
                            ],
                            justify="between",
                            style={"margin-bottom": "20px"},
                        ),

                        html.Br(),
                        dash_table.DataTable(
                                id="search-results-table",
                                columns=[{"name": col, "id": col} for col in matches.columns],
                                data=matches.to_dict("records"),

                                editable=False,
                                # filter_action="native",

                                sort_action="native",
                                sort_mode="multi",
                                
                                column_selectable=False,
                                row_selectable=False,
                                row_deletable=False,
                                
                                selected_columns=[],
                                selected_rows=[],
                                
                                page_action="native",
                                page_current= 0,
                                page_size= 20,
                                style_table={'width': '900px'},
                                style_header={'fontWeight': 'bold'},
                                style_cell={
                                    'textAlign': 'left',
                                    'fontSize': '14px',
                                    'verticalAlign': 'top',
                                    'whiteSpace': 'pre-line'
                                },
                                style_cell_conditional=[
                                    {'if': {'column_id': 'Text (Paragraph)'},
                                    'width': '1000px'},
                                ],
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(250, 250, 250)',
                                    }
                                ],
                                style_as_list_view=True,
                            )
                        ]
            )



#################################################
# end of function page
#################################################






#################################################
@app.callback(
    Output("collapse", "is_open"),
    [Input("toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(port=8888, debug=True)