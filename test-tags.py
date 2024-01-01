import pandas as pd
import dash
from dash import html, dcc
import dash_table
from dash.dependencies import Input, Output, State

# Sample DataFrame
df = pd.DataFrame({
    'Tag': ['tag1', 'tag2', 'tag1', 'tag3', 'tag2', 'tag3'],
    'Value': ['A', 'B', 'C', 'D', 'E', 'F']
})



# Unique tags from the DataFrame
tags = df['Tag'].unique()

tags = [{'Tag': 'tag1', 'Value': 'A'},
 {'Tag': 'tag2', 'Value': 'B'},
 {'Tag': 'tag3', 'Value': 'D'},]


tags = {
    'tag111':'tag1',
    'tag222':'tag2',
    'tag333':'tag3',
}

app = dash.Dash(__name__)

app.layout = html.Div([
    # html.Div(id='tag-container', children=[html.Button(tag, id={'type': 'tag', 'index': i}) for i, tag in enumerate(tags)]),
    html.Div(id='tag-container', children=[html.Button(key, id={'type': 'tag', 'index': i}) for i, key in enumerate(tags)]),
    # dash_table.DataTable(id='table', columns=[{"name": i, "id": i} for i in df.columns], data=df.to_dict('records'))
    dcc.Loading(id="loading3", type="default", children=html.Div(id="search-results3"), fullscreen=False),

])

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
    print(tags[tag_clicked])
    filtered_df = df[df['Tag'] == tags[tag_clicked]]
    print(filtered_df)
    # return filtered_df.to_dict('records')
    return  html.Div(children=[
        dash_table.DataTable(id='table', 
                             columns=[{"name": i, "id": i} for i in filtered_df.columns], 
                             data=filtered_df.to_dict('records'),
                            #  data=df.to_dict('records')
                             )
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
