import dash
import dash_table
from dash import html

app = dash.Dash(__name__)

data=[
    {"name": "328", "url": "[328](https://docs.wto.org/dol2fe/Pages/FE_Search/FE_S_S006.aspx?SymbolList=WT/TPR/S/328/Rev.1)"},
    {"name": "OpenAI", "url": "[OpenAI](https://www.openai.com)"},
]

app.layout = html.Div([
    dash_table.DataTable(
        id='table',
        columns=[
            {"name": i, "id": i, 'type': 'text', 'presentation': 'markdown'} for i in data[0].keys()
        ],
        data=data
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)


# https://docs.wto.org/dol2fe/Pages/FE_Search/FE_S_S006.aspx?SymbolList=WT/TPR/S/328/Rev.1
