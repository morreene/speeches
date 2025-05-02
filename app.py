from flask import Flask, session
from flask_session import Session
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import requests

#################################################
#####     configurations
#################################################

# Function to get the server's public IP address
def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        if response.status_code == 200:
            return response.json()['ip']
        return "Could not determine IP"
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return "Error getting IP"

# Get the public IP address once at startup
PUBLIC_IP = get_public_ip()
print(f"Application public IP: {PUBLIC_IP}")

client = AzureOpenAI(
  api_key = "deca3c66de3649338b35ab92c04ba309",  
  api_version = "2023-05-15",
  azure_endpoint ="https://oaishrp02.openai.azure.com/" 
)

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

# Try to load the parquet file, with error handling for Heroku deployment
try:
    speechdb = pd.read_parquet('data/speech-text-embedding-20240508.parquet')
    contextdb = speechdb[speechdb['n_tokens']>50].copy()
    speechlist = speechdb.groupby(['Subfolder','FileName']).size().reset_index(name='NParas')
    speechlist.columns = ['Folder','File Name','Number of paragraphs']
    print("Successfully loaded parquet file")
except FileNotFoundError:
    print("Warning: Could not find the parquet file. Using empty dataframes for demo/development purposes.")
    # Create empty dataframes with the necessary structure for development/demo
    speechdb = pd.DataFrame(columns=['Subfolder', 'FileName', 'ParagraphID', 'Text', 'n_tokens', 'ada_v2'])
    contextdb = speechdb.copy()
    speechlist = pd.DataFrame(columns=['Folder', 'File Name', 'Number of paragraphs'])

#################################################
##### Speech app
#################################################

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002-yu-7jo4bn"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# search through the reviews for a specific product
def search_speech_db(df, user_query, ncontext=20):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002-yu-7jo4bn" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
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
    response = client.chat.completions.create(
        model=model,
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
USERS = {"admin": "admin", "ersd": "wtr2024", "ierd": "ierd"}

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
    html.A([dbc.Col(html.Img(src=app.get_asset_url("logo.png"),  width="180px", style={'margin-left':'15px', 'margin-bottom':'50px'}))], href="/page-2"),
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
                                # dbc.NavLink("Write", href="/page-1", id="page-1-link", style={'display': 'block' if session.get('username') == 'admin' else 'none'}),
                                dbc.NavLink("Search", href="/page-2", id="page-2-link"),
                                dbc.NavLink("Browse by topics", href="/page-3", id="page-3-link"),
                                dbc.NavLink("Speech List", href="/page-4", id="page-4-link"),
                                dbc.NavLink("About", href="/page-5", id="page-5-link"),
                                dbc.NavLink("Logout", href="/logout", active="exact"),  # Add a logout link
                            ], vertical=True, pills=False,
                        ), id="collapse",
                    ),
                    html.Div([html.P("V0.3 (20240511)",
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
    dcc.Location(id='url', refresh=True),
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
                            html.Hr(),
                            html.Div([
                                html.P(f"App Public IP: {PUBLIC_IP}", style={"color": "gray", "font-size": "12px"}),
                            ]),
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
    [Input('login-button', 'n_clicks'),
     Input('url', 'pathname')],
    [State('username', 'value'), State('password', 'value')]
)
def update_output(n_clicks, pathname, username, password):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'url' and pathname == '/logout':
        # Handle logout
        session.clear()
        return {}, {'display': 'none'}
    
    if trigger_id == 'login-button' and n_clicks > 0:
        if username in USERS and USERS[username] == password:
            session['authed'] = True
            session['username'] = username  # Store username in session
    
    if session.get('authed', False):
        return {'display': 'none'}, {'display': 'block'}
    else:
        return {}, {'display': 'none'}



@app.callback(
    Output("page-1-link", "style"),
    [Input("url", "pathname")])  # Trigger this callback whenever the URL changes
def toggle_write_link_visibility(pathname):
    if session.get('authed') and session.get('username') in ['admin', 'ersd']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# render content according to path
@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/logout":
        # Just return an empty div, the other callback will handle the logout
        return html.Div()
        
    # elif pathname in ["/","/login", "/page-1"]:
    elif pathname == "/page-1":
        return html.Div([
            html.H4("Draft Speech Using Guidelines & Background from Past Speeches and Knowledge Base (KB)", ),
            html.Br(),
            html.H6("Enter Topics & Keywords for Speech Content and KB Searches: [Required]"),
            dbc.Row([
                dbc.Col(
                    dbc.InputGroup([
                            dbc.Input(id="write-input-box", type="text", placeholder="Enter a topic: e.g. globalization OR digital trade"),
                    ])
                )], justify="center", className="header", id='search-container1',
            ),

            html.Br(),
            html.H6("Specify Additional Requirements, Background, and Outlines:"),
            dbc.Row([
                dbc.Col(
                    dbc.Textarea(id="write-textarea-additional",  placeholder="Enter Additional Requirements, Background, and Outlines:", size="md",rows=4, style={"width": "100%"})
                )
                ], justify="center", className="header", id='search-container2', 
            ),
            dbc.Row(
                [
                    dbc.Col(html.H6(["Temperature (creativity):"]), 
                                    width=3,  style={'margin-top':5,'margin-left':0}),
                    dbc.Col(
                        dcc.Slider(0, 1, 0.2, value=0.4, id='write-slider-temperature'),style={"margin-top": "20px"}, width=3,
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
                ], align="center", style={"margin-bottom": "0px"}),

            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.H6('Audience and style: '), width=3, style={'margin-top':5,'margin-left':0}),                    
                    dbc.Col(
                        [
                            # html.H6('Audience and style'),
                            dcc.Dropdown(
                                id='write-dropdown-style',
                                multi=False,
                                options=[{'label': i[0], 'value': i[1]} for i in styles.items()],
                                value='speak to delegates/heads of states: Diplomatic, formal, strategic, respectful, authoritative, policy-oriented, persuasive, factual, concise, collaborative',
                                clearable=False
                            ),
                        ], width=3,
                    )                    
                ],
                align="center",
                style={"margin-bottom": "10px"},
            ),


            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.H6('[Model to use:]'), width=2, style={'margin-top':5,'margin-left':0}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="write-radio-select-model",
                            options=[
                                {"label": 'ChatGPT 3.5 Turbo 16k', "value": 'gpt-35-turbo-16k'},
                                {"label": 'ChatGPT 4', "value": 'gpt-4'},
                            ],
                            value='gpt-35-turbo-16k',
                            inline=True,
                        ),
                        width=4,
                    ),


                    dbc.Col(html.H6('[Number of paras from KB as inputs:]'), width=4, style={'margin-top':2,'margin-left':0}),
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
                        width=2,
                    ),
                ],
                align="center",
                style={"margin-bottom": "0px"},
            ),

            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Draft speech ...", id="write-submit-button", n_clicks=0, color='info'), width={'size': 12}, className='text-right'
                        ),
                ],
                align="right",
            ),


            html.Hr(),
            dbc.Row([
                dbc.Col(html.H6("Sample topic: ", className='text-left'),  width=12),
                dbc.Row([
                    dbc.Col(
                        dcc.Markdown('''
                            - Trade and environment
                            - Globalization and re-globalization
                            - WTO and multilateral trading system
                            - US and China trade war
                            - Industrial policy                                     
                        '''),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Markdown('''
                            - Subsidies
                            - Least developed country and trade
                            - Africa and trade
                            - Digital trade
                            - Davos Ministerial Conference, MC13, Growth, Transportation, Fragmentation, Uncertainty, Green trade, Service, Trade resilience.
                        '''),
                        width=6
                    ),
                ]),
            ], justify="center", #className="header", 
            id='write-sample-topics'),

            html.Br(),
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        id="loading2", 
                        type="default", 
                        children=html.Div(id="write-results"), 
                        fullscreen=False,
                        style={"position":"absolute", "left":"300px", "top":"20px"}
                    ),
                    width=12
                ),
            ], justify="center")



        ])

    # Set "Search" as the home page
    # elif pathname == "/page-2":    
    elif pathname in ["/","/login", "/page-2"]:
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
        ])
    
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
        ])

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
            ])

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
                            html.Hr(),
                            html.Div([
                                html.H6("Application Information:", style={"margin-top": "20px"}),
                                html.P(f"Public IP Address: {PUBLIC_IP}", style={"color": "blue", "font-weight": "bold"}),
                                html.P("Note: Add this IP to your Azure OpenAI firewall allowed list if you're experiencing access issues.")
                            ])
                ])

    else:
        return html.P("404: Not found")




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
        try:
            # ncontext = 20
            # audience = 'delegates to the WTO'
            # model="gpt-4"
            # topic = 'reglobalization'
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
        except Exception as e:
            print(f"Write draft error: {e}")
            return html.Div(
                        dbc.Container(
                            [
                                dbc.Row(
                                    [html.P(f"An error occurred while generating the speech. This could be because the database is not loaded or accessible. Details: {str(e)}")],
                                    justify="between",
                                    style={"margin-bottom": "5px"},
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
        try:
            df = search_speech_db(speechdb, search_terms, ncontext=top)
            if len(df) == 0:
                return html.Div(html.P("No results found or data not available. This could be because the database is not loaded.")), {'display': 'none'}
                
            df['meta'] = df['FileName'] + '\n Para: ' + df['ParagraphID'].astype(str) + '\n Score: ' + df['similarities'].astype(str) 
            df['text'] = df['Text']

            matches = df[['meta', 'text']]
            matches.columns = ['Meta','Text (Paragraph)']

            # Display the results in a datatable
            return html.Div(style={'width': '100%'},
                        children=[
                            html.Br(),
                            dbc.Row(
                                [
                                    # dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['similarities'].min()) + ' to ' + str(df['similarities'].max())), width={"size": 9, "offset": 0}),
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
                ),  {'display': 'none'}
        except Exception as e:
            print(f"Search error: {e}")
            return html.Div(html.P(f"An error occurred during search. This could be because the database is not loaded or accessible.")), {'display': 'none'}

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

    try:
        df = search_speech_db(speechdb, tags[tag_clicked], ncontext=50)
        if len(df) == 0:
            return html.Div(html.P("No results found or data not available. This could be because the database is not loaded."))
            
        df['meta'] = df['FileName'] + '\n Para: ' + df['ParagraphID'].astype(str) + '\n Score: ' + df['similarities'].astype(str) 
        df['text'] = df['Text']

        matches = df[['meta', 'text']]
        matches.columns = ['Meta','Text (Paragraph)']

        # Display the results in a datatable
        return html.Div(style={'width': '100%'},
                    children=[
                        html.Br(),
                        dbc.Row(
                            [
                                # dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['similarities'].min()) + ' to ' + str(df['similarities'].max())), width={"size": 9, "offset": 0}),
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
    except Exception as e:
        print(f"Topic search error: {e}")
        return html.Div(html.P(f"An error occurred during topic search. This could be because the database is not loaded or accessible."))

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