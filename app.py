from flask import Flask, session
from flask_session import Session
from dash import Dash, dcc, html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
# to delete dash_table
from dash import html, dcc, dash_table
import urllib.parse

import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pinecone


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

with open('data/about.md', 'r') as markdown_file:
    markdown_about = markdown_file.read()

# matrix = pd.read_pickle("data/tpr_matrix.pickle")
# matrix.index.name = None

# member_list = pd.read_pickle("data/all_mem.pickle")
# member_list = member_list['Member'].tolist()
# member_list = ['All Members'] + member_list

# cat_list = pd.read_pickle("data/all_cat.pickle")
# cat_list = cat_list['Topic'].tolist()
# cat_list = ['All topics (slow loading)'] + cat_list

# tags for topic keywords
tags = {
    'Africa':               'Africa trade African Continental Free Trade Area (AfCFTA)',
    'COVID-19':               'covid vaccine',
    'Digital trade':          'digital trade ecommerce moratorium on electronic transmissions',
    'E-commerce':            'ecommerce',
    'Environment':          'environment climate polution environmental protection',
    'Geopolitics':               'geopolitics US-China geopolitical',
    'Global economy':               'global economy GDP growth trend',
    'Intellectual property': 'intellectual property rights',
    'MSME':                 'micro small and medium enterprises',
    'Subsidies':            'industrial subsidies grant',
}

speechdb = pd.read_parquet('data/speech-text-embedding.parquet')
matrix = speechdb.groupby(['Subfolder','FileName']).size().reset_index(name='NParas')




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
    res = search_speech_db(speechdb, topic, ncontext=ncontext)
    return res['Text'].to_list()

def build_prompt_with_context(topic, context=[], nwords=300, audience='government officials'):
    return [{'role': 'system', 
             'content': f'''As a speech writer, you are tasked with composing a speech for the Director General of the World Trade Organization. \
                            The speech should address the specific topic provided by the user, incorporating relevant contexts and information as mentioned. \
                            Ensure that the speech mirrors the style of the context given, adhering to the specified tone - be it formal, persuasive, informative, or any other. \
                            Additionally, the speech should be tailored to meet the exact length requirement set by the user, specified in the number of words. \
                            Your goal is to craft a speech that effectively conveys the WTO's perspective on the given topic, while maintaining the Director General's tone and style.'''
                    }, 
            {'role': 'user', 
             'content': f"""
                        Topic:
                        {topic}       
                            
                        The context is the following:
                        {' '.join(context)}

                        Adjust the contents and tone for targeted audience:
                        {audience}

                        Length:
                        {nwords} words

                        Speech:
            """}]

def write_speech(message, temperature=0, model="gpt-35-turbo"):
    response = openai.ChatCompletion.create(
        engine=model,
        messages=message,
        temperature=temperature,
        max_tokens=1000,
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
        #    external_stylesheets=[dbc.themes.BOOTSTRAP], 
           external_stylesheets = external_stylesheets,
           suppress_callback_exceptions=True
           )

app.title = 'Speech Database'
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-62289743-10"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'UA-62289743-10');
        </script>

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
                    html.Div([html.P("V0.2 (20240111)",
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
        return dcc.Location(pathname="/login", id="redirect-to-login"), "/logout"
    # elif pathname == "/":
    #     # return html.P("This is the content of the home page!"), pathname
    #     # return dcc.Location(pathname="/page-1", id="redirect-to-login"), "/page-1"  
    #     return dcc.Location(pathname="/page-1", id="redirect-to-login"), "/page-1"      
    # elif pathname == "/login":
    #     # return html.P("This is the content of page after login"), pathname
    #     return dcc.Location(pathname="/page-1", id="redirect-to-login"), "/page-1"
    # elif pathname == "/page-1":
    elif pathname in ["/","/login", "/page-1"]:




















        return dbc.Container([
            html.H6("Write based on the previous speeches", className="display-about"),
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Col(
                        dbc.InputGroup([
                                dbc.Input(id="search-box2", type="text", placeholder="Topics: e.g. globalization and reglobalization"),
                                dbc.Button(" Write about ...", id="search-button2", n_clicks=0),
                            ]
                        ), width=12,
                    ),
                ], justify="center", className="header", id='search-container2'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(html.Label('Number of relevent paragraphs as input: '), width="auto", style={'margin-top':5,'margin-left':10}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="radio-select-top2",
                            options=[
                                {"label": '20', "value": 20},
                                {"label": '30', "value": 30},
                            ],
                            value=20,
                            inline=True,
                        ),
                        width=True,
                    ),
                ],
                align="center",
                style={"margin-bottom": "10px"},
            ),

            dbc.Row(
                [
                    dbc.Col(html.Label("Length of the draft:"), width="auto",  style={'margin-top':5,'margin-left':10}),
                    dbc.Col(
                        dbc.RadioItems(
                            id="radio-select-words",
                            options=[
                                {"label": "200 words", "value": 200},
                                {"label": "300 words", "value": 300},
                                {"label": "500 words", "value": 500},
                            ],
                            value=300,
                            inline=True,
                        ),
                        width=True,
                    ),
                ],
                align="center",
                style={"margin-bottom": "10px"},
            ),

            dbc.Row(
                [
                    dbc.Col(html.P("Degree of randomness (temperature): Increase for more creativity"), width="auto",  style={'margin-top':5,'margin-left':10}),
                    dbc.Col(
                        dcc.Slider(0, 1, 0.1,
                                value=0.5,
                                id='slider-temperature',
                        ),style={"margin-top": "20px"},
                    ),
                ],
                align="center",
                style={"margin-bottom": "1px"},
            ),


            html.Hr(),
            # html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown('''
                                Sample topics:
                                - Trade and environment
                                - Globalization and re-globalization
                                - WTO and multilateral trading system
                                - US and China trade war
                                - Trade finance
                                - Aid for trade 
                                - Least developed country members
                                - Africa and global trade
                                - Digital trade '''
                        ),
                ], width=12),
            ], justify="center", className="header", id='sample-queries2'),
            html.Br(),
            html.Br(),
            dbc.Row([ 
                # html.Div(id="search-results", className="results"),
                dbc.Col([
                        dcc.Loading(id="loading2", type="default", children=html.Div(id="search-results2"), fullscreen=False),
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
                                dbc.Input(id="search-box", type="text", placeholder="Enter search query, e.g. subsidies and government support to fossil feul and energy", ),
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
                    columns=[{"name": i, "id": i} for i in matrix.columns],
                    data=matrix.to_dict('records'),
                    style_cell_conditional=[
                        {'if': {'column_id': 'Member'},
                         'width': '100px'},
                    ]
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
#####     Page Chat
#################################################

# call back for returning results
@app.callback(
        [Output("search-results2", "children"),  
         Output("sample-queries2", "style")
        ],
        [Input("search-button2", "n_clicks"),
         Input("search-box2", "n_submit")
        ], 
        [State("search-box2", "value"),
         State('radio-select-top2', 'value'),
         State('radio-select-words', 'value'),
         State('slider-temperature', 'value')
        ]
        )
def chat(n_clicks, n_submit, topic, ncontext, nwords, temperature):
    # Check if the search button was clicked
    # if (n_clicks <=0 and n_submit is None) or search_terms=='' or search_terms is None:
    if (n_clicks <=0 and n_submit is None) or topic=='' or topic is None:
        return "",  None
    else:
        
        audience = 'delegates to the WTO'
        model="gpt-4"
        # topic = 'reglobalization'
        # topic = 'trade under most favoriate nation principle'
        # topic = 'ecommerce'
        # topic = 'trade in Africa'
        # topic = 'Trade and environment'
        # topic = "China and US trade war"
        # topic = topic
        # ncontext = 20
        context = generate_context(topic, ncontext)

        # nwords = 300
        message = build_prompt_with_context(topic, context, nwords, audience)

        # temperature = 0
        draft = write_speech(message, temperature, model=model)

    return html.Div(
    dbc.Container(
        [
            dbc.Row(
                [html.P('Draft (' + str(len(draft.split()))  +" words): ")],
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
#####     Page tags
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