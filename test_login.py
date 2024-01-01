from flask import Flask, session
from flask_session import Session
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Hardcoded users (for demo purposes)
USERS = {"admin": "admin"}

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecretkey'
server.config['SESSION_TYPE'] = 'filesystem'

Session(server)

app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Input(id='username', placeholder='username', type='text'),
    dbc.Input(id='password', placeholder='password', type='password'),
    dbc.Button(id='login-button', children='Login', n_clicks=0),
    html.Div([sidebar, content], id='page-layout', style={'display': 'none'}),
])

@app.callback(
    Output('username', 'style'),
    Output('password', 'style'),
    Output('login-button', 'style'),
    Output('page-layout', 'style'),
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value')]
)
def update_output(n_clicks, username, password):
    if n_clicks > 0:
        if username in USERS and USERS[username] == password:
            session['authed'] = True
    if session.get('authed', False):
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {}, {}, {}, {'display': 'none'}

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(port=8888)






















































# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc

# # Hardcoded users (for demo purposes)
# USERS = {"admin": "admin"}

# # the style arguments for the sidebar. We use position:fixed and a fixed width
# SIDEBAR_STYLE = {
#     "position": "fixed",
#     "top": 0,
#     "left": 0,
#     "bottom": 0,
#     "width": "16rem",
#     "padding": "2rem 1rem",
#     "background-color": "#f8f9fa",
# }

# # the styles for the main content position it to the right of the sidebar and
# # add some padding.
# CONTENT_STYLE = {
#     "margin-left": "18rem",
#     "margin-right": "2rem",
#     "padding": "2rem 1rem",
# }

# sidebar = html.Div(
#     [
#         html.H2("Sidebar", className="display-4"),
#         html.Hr(),
#         html.P(
#             "A simple sidebar layout with navigation links", className="lead"
#         ),
#         dbc.Nav(
#             [
#                 dbc.NavLink("Home", href="/", active="exact"),
#                 dbc.NavLink("Page 1", href="/page-1", active="exact"),
#                 dbc.NavLink("Page 2", href="/page-2", active="exact"),
#             ],
#             vertical=True,
#             pills=True,
#         ),
#     ],
#     style=SIDEBAR_STYLE,
# )

# content = html.Div(id="page-content", style=CONTENT_STYLE)

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     dbc.Input(id='username', placeholder='username', type='text'),
#     dbc.Input(id='password', placeholder='password', type='password'),
#     dbc.Button(id='login-button', children='Login', n_clicks=0),
#     html.Div([sidebar, content], id='page-layout', style={'display': 'none'}),
# ])

# @app.callback(
#     Output("page-content", "children"),
#     [Input("url", "pathname")])
# def render_page_content(pathname):
#     if pathname == "/":
#         return html.P("This is the content of the home page!")
#     elif pathname == "/page-1":
#         return html.P("This is the content of page 1. Yay!")
#     elif pathname == "/page-2":
#         return html.P("Oh cool, this is page 2!")
#     # If the user tries to reach a different page, return a 404 message
#     return dbc.Jumbotron(
#         [
#             html.H1("404: Not found", className="text-danger"),
#             html.Hr(),
#             html.P(f"The pathname {pathname} was not recognised..."),
#         ]
#     )

# @app.callback(
#     Output('username', 'style'),
#     Output('password', 'style'),
#     Output('login-button', 'style'),
#     Output('page-layout', 'style'),
#     [Input('login-button', 'n_clicks')],
#     [State('username', 'value'), State('password', 'value')]
# )
# def update_output(n_clicks, username, password):
#     if n_clicks > 0:
#         if username in USERS and USERS[username] == password:
#             return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
#         else:
#             return {}, {}, {}, {'display': 'none'}
#     else:
#         return {}, {}, {}, {'display': 'none'}

# if __name__ == "__main__":
#     app.run_server(port=8888)








































# import dash
# from dash import html, dcc, callback
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc

# # Style
# CONTENT_STYLE = {"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"}

# # Simple username and password check
# def check_password(username, password):
#     return username == 'w' and password == 'w'

# # UI for Login
# login_form = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             dbc.Row([
#                 dbc.Label('Username:'),
#                 dbc.Input(id='login-username', type='text')
#             ]),
#             dbc.Row([
#                 dbc.Label('Password:'),
#                 dbc.Input(id='login-password', type='password')
#             ]),
#             dbc.Button('Login', id='login-button', color='primary')
#         ])
#     ])
# ], className='mt-5')

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='content'),
#     html.Div(id='hidden-div', style={'display': 'none'})
# ])

# @app.callback(
#     Output('content', 'children'),
#     Input('url', 'pathname'),
#     Input('hidden-div', 'children')
# )
# def render_page_content(pathname, login_status):
#     if pathname == '/':
#         if login_status == 'Logged in':
#             return html.Div("This is the content of the home page!", style=CONTENT_STYLE)
#         else:
#             return login_form
#     elif pathname == '/page-1':
#         if login_status == 'Logged in':
#             return html.Div("This is the content of page 1. Yay!", style=CONTENT_STYLE)
#         else:
#             return login_form
#     elif pathname == '/page-2':
#         if login_status == 'Logged in':
#             return html.Div("Oh cool, this is page 2!", style=CONTENT_STYLE)
#         else:
#             return login_form

# @app.callback(
#     Output('hidden-div', 'children'),
#     Input('login-button', 'n_clicks'),
#     State('login-username', 'value'),
#     State('login-password', 'value')
# )
# def update_output(n_clicks, input1, input2):
#     if n_clicks is None:
#         return 'Not logged in'
#     else:
#         if check_password(input1, input2):
#             return 'Logged in'
#         else:
#             return 'Not logged in'

# if __name__ == "__main__":
#     app.run_server(port=8888, debug=True)
