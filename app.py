from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text, export_graphviz
import io, base64
import pydotplus
import pickle

# Datos
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

app = Dash(__name__, suppress_callback_exceptions=True)

# Layout principal
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    children=dmc.AppShell(
        id="app-shell",
        padding="md",
        navbarOffsetBreakpoint="sm",
        asideOffsetBreakpoint="sm",
        navbar=dmc.Navbar(
            id="navbar",
            p="xs",
            hiddenBreakpoint="sm",
            hidden=True,
            width={"base": 250, "sm": 250},
            children=[
                dmc.NavLink(label="Inicio", href="/", active=True),
                dmc.NavLink(label="Tabla", href="/table"),
                dmc.NavLink(label="Gráfico", href="/graph"),
                dmc.NavLink(label="Bubble Plot", href="/bubble"),
                dmc.NavLink(label="Distribuciones", href="/distribution"),
                dmc.NavLink(label="Scatter Plot", href="/scatter"),
                dmc.NavLink(label="Regresión", href="/regresion"),
                dmc.NavLink(label="Árbol de Decisión", href="/arbol"),
            ]
        ),
        header=dmc.Header(
            height=60,
            p="xs",
            children=dmc.Group([
                dmc.MediaQuery(
                    dmc.Burger(id="burger", opened=False, size="sm"),
                    smallerThan="sm",
                    styles={"display": "block"}
                ),
                dmc.Title("Gapminder Dashboard", order=3)
            ])
        ),
        footer=dmc.Footer(height=40, p="md", children="© 2025 Dashboard Pro"),
        children=[
            dcc.Location(id="url"),
            html.Div(id="page-content")
        ]
    )
)

# Layouts
inicio_layout = dmc.Container([
    dmc.Title("Dashboard Profesional - Gapminder 2007", align="center", color="blue"),
    dmc.Space(h=20),
    dmc.Grid([
        dmc.Col(dmc.Paper(
            dmc.Stack([
                dmc.Text("Esperanza de vida promedio", weight=700),
                dmc.Text(f"{df['lifeExp'].mean():.2f} años", size="xl")
            ]), p="md", shadow="sm"), span=4),
        dmc.Col(dmc.Paper(
            dmc.Stack([
                dmc.Text("Población total", weight=700),
                dmc.Text(f"{df['pop'].sum()/1e9:.2f} B", size="xl")
            ]), p="md", shadow="sm"), span=4),
        dmc.Col(dmc.Paper(
            dmc.Stack([
                dmc.Text("PIB per cápita promedio", weight=700),
                dmc.Text(f"${df['gdpPercap'].mean():,.0f}", size="xl")
            ]), p="md", shadow="sm"), span=4),
    ], gutter="md"),
    dmc.Space(h=30),
    dmc.Text("Use el menú lateral para explorar los datos por sección", size="md")
])

table_layout = dmc.Container([
    dmc.Title("Tabla de Datos"),
    dash_table.DataTable(data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'})
])

graph_layout = dmc.Container([
    dmc.Title("Gráfico Interactivo"),
    dmc.Select(label="Variable", id="var-select", data=[{"label": i, "value": i} for i in ['pop', 'lifeExp', 'gdpPercap']], value="lifeExp"),
    dcc.Graph(id="main-graph")
])

bubble_layout = dmc.Container([
    dmc.Title("Bubble Plot"),
    dcc.Graph(figure=px.scatter(df, x='gdpPercap', y='lifeExp', size='pop', color='continent', hover_name='country', log_x=True))
])

distribution_layout = dmc.Container([
    dmc.Title("Distribuciones por País"),
    dmc.Select(label="Seleccione un país", id="country-select", data=[{"label": c, "value": c} for c in df.country], value="Canada"),
    dmc.Grid([
        dmc.Col(dcc.Graph(id='dist-lifeExp'), span=4),
        dmc.Col(dcc.Graph(id='dist-pop'), span=4),
        dmc.Col(dcc.Graph(id='dist-gdp'), span=4),
    ])
])

scatter_layout = dmc.Container([
    dmc.Title("Scatter Plot Interactivo"),
    dmc.Select(label="Seleccione continente", id="continent-select",
               data=[{"label": c, "value": c} for c in df.continent.unique()] + [{"label": "Todos", "value": "Todos"}],
               value="Todos"),
    dcc.Graph(id="scatter-plot")
])

regression_layout = dmc.Container([
    dmc.Title("Modelo de Regresión Lineal Múltiple", order=2),
    dmc.Text("Predicción de la esperanza de vida usando variables seleccionadas."),
    dmc.MultiSelect(
        id="reg-vars",
        label="Selecciona las variables predictoras",
        data=[
            {"label": "Población (pop)", "value": "pop"},
            {"label": "PIB per cápita (gdpPercap)", "value": "gdpPercap"},
            {"label": "Continente (continent)", "value": "continent"},
        ],
        value=["pop", "gdpPercap", "continent"]
    ),
    dmc.Button("Entrenar Modelo", id="train-btn"),
    dmc.Space(h=20),
    html.Div(id="model-output"),
    dcc.Graph(id="regression-graph"),
    dmc.Space(h=20),
    html.Div(id="coeff-table"),
    html.A("Descargar modelo entrenado", id="download-link", href="", download="modelo_regresion.pkl", style={"display": "none"})
])

tree_layout = dmc.Container([
    dmc.Title("Modelo de Árbol de Decisión", order=2),
    dmc.Text("Predicción de la esperanza de vida usando árboles de decisión."),
    dmc.MultiSelect(
        id="tree-vars",
        label="Selecciona las variables predictoras",
        data=[
            {"label": "Población (pop)", "value": "pop"},
            {"label": "PIB per cápita (gdpPercap)", "value": "gdpPercap"},
            {"label": "Continente (continent)", "value": "continent"},
        ],
        value=["pop", "gdpPercap", "continent"]
    ),
    dmc.NumberInput(
        id="tree-depth",
        label="Profundidad máxima del árbol (max_depth)",
        min=1,
        max=10,
        step=1,
        value=3,
        style={"maxWidth": "300px"}
    ),
    dmc.Button("Entrenar Árbol", id="tree-train-btn"),
    dmc.Space(h=20),
    html.Div(id="tree-text-output"),
    html.Img(id="tree-image", style={"width": "100%", "maxWidth": "800px"})
])

# Routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    return {
        "/": inicio_layout,
        "/table": table_layout,
        "/graph": graph_layout,
        "/bubble": bubble_layout,
        "/distribution": distribution_layout,
        "/scatter": scatter_layout,
        "/regresion": regression_layout,
        "/arbol": tree_layout
    }.get(path, inicio_layout)

@app.callback(Output("navbar", "hidden"), Input("burger", "opened"), prevent_initial_call=True)
def toggle_navbar(opened):
    return not opened

@app.callback(Output("main-graph", "figure"), Input("var-select", "value"))
def update_main_graph(var):
    return px.histogram(df, x='continent', y=var, histfunc='avg')

@app.callback(
    Output("dist-lifeExp", "figure"),
    Output("dist-pop", "figure"),
    Output("dist-gdp", "figure"),
    Input("country-select", "value")
)
def update_distributions(country):
    figs = []
    for col in ["lifeExp", "pop", "gdpPercap"]:
        fig = px.histogram(df, x=col, nbins=20, title=f"Distribución de {col}")
        val = df[df.country == country][col].values[0]
        fig.add_vline(x=val, line_color="red")
        figs.append(fig)
    return figs

@app.callback(Output("scatter-plot", "figure"), Input("continent-select", "value"))
def update_scatter(cont):
    data = df if cont == "Todos" else df[df.continent == cont]
    return px.scatter(data, x='gdpPercap', y='lifeExp', size='pop', color='continent', hover_name='country')

@app.callback(
    Output("model-output", "children"),
    Output("regression-graph", "figure"),
    Output("coeff-table", "children"),
    Output("download-link", "href"),
    Output("download-link", "style"),
    Input("train-btn", "n_clicks"),
    State("reg-vars", "value"),
    prevent_initial_call=True
)
def entrenar_modelo(n_clicks, variables):
    if not variables:
        return dmc.Text("Seleccione al menos una variable."), {}, "", "", {"display": "none"}

    X = df[variables].copy()
    y = df["lifeExp"]
    if "continent" in variables:
        transformers = [("cat", OneHotEncoder(drop="first"), ["continent"])]
    else:
        transformers = []

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    output_text = [
        dmc.Title("Resultados del Modelo", order=3),
        dmc.Text(f"R² Score: {r2:.3f}"),
        dmc.Text(f"Error cuadrático medio (MSE): {mse:.2f}")
    ]

    feature_names = []
    if "continent" in variables:
        ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
        feature_names.extend(ohe.get_feature_names_out(["continent"]))
    feature_names.extend([v for v in variables if v != "continent"])

    coefs = model.named_steps["regressor"].coef_
    coef_table = dash_table.DataTable(
        columns=[{"name": "Variable", "id": "var"}, {"name": "Coeficiente", "id": "coef"}],
        data=[{"var": name, "coef": f"{val:.4f}"} for name, val in zip(feature_names, coefs)]
    )

    fig = px.scatter(x=y, y=y_pred, labels={"x": "Real", "y": "Predicho"}, title="Comparación: Real vs Predicho")
    fig.add_trace(px.line(x=y, y=y, labels={"x": "Real", "y": "Predicho"}).data[0])

    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64_model = base64.b64encode(buffer.read()).decode()

    return output_text, fig, coef_table, f"data:application/octet-stream;base64,{b64_model}", {"display": "block"}

@app.callback(
    Output("tree-text-output", "children"),
    Output("tree-image", "src"),
    Input("tree-train-btn", "n_clicks"),
    State("tree-vars", "value"),
    State("tree-depth", "value"),
    prevent_initial_call=True
)
def entrenar_arbol(n_clicks, variables, max_depth):
    if not variables:
        return dmc.Text("Seleccione al menos una variable."), ""

    X = df[variables].copy()
    y = df["lifeExp"]

    if "continent" in variables:
        X = pd.get_dummies(X, columns=["continent"], drop_first=True)

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    model.fit(X, y)

    reglas = export_text(model, feature_names=list(X.columns))
    dot_data = io.StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X.columns,
                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    img = graph.create_png()
    b64_img = base64.b64encode(img).decode()

    return html.Pre(reglas), f"data:image/png;base64,{b64_img}"

# Ejecutar en modo Colab
# Ejecutar la app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
