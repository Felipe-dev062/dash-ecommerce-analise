import pandas as pd
import numpy as np
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Leitura e preparacao
# =========================
ARQUIVO_CSV = 'ecommerce_estatistica.csv'

df = pd.read_csv(ARQUIVO_CSV)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Renomeia colunas com acento para facilitar o codigo
rename_map = {
    'Título': 'Titulo',
    'N_Avaliações': 'N_Avaliacoes',
    'Gênero': 'Genero',
    'Preço': 'Preco',
    'Preço_MinMax': 'Preco_MinMax',
}
df = df.rename(columns=rename_map)

# Remove nulos para evitar erros nos graficos
plot_df = df.dropna().copy()

# =========================
# Graficos
# =========================

# 1) Histograma - distribuicao de preco
fig_hist = px.histogram(
    plot_df,
    x='Preco',
    nbins=30,
    title='Histograma do Preco dos Produtos',
    labels={'Preco': 'Preco', 'count': 'Quantidade'}
)
fig_hist.update_layout(bargap=0.08)

# 2) Dispersao - preco x nota
fig_scatter = px.scatter(
    plot_df,
    x='Preco',
    y='Nota',
    color='Desconto',
    size='N_Avaliacoes',
    hover_data=['Marca', 'Material', 'Temporada'],
    title='Grafico de Dispersao: Preco x Nota',
    labels={
        'Preco': 'Preco',
        'Nota': 'Nota',
        'Desconto': 'Desconto',
        'N_Avaliacoes': 'Numero de Avaliacoes'
    }
)

# 3) Mapa de calor - correlacao
colunas_numericas = [
    'Nota', 'N_Avaliacoes', 'Desconto', 'Preco', 'Nota_MinMax',
    'N_Avaliacoes_MinMax', 'Desconto_MinMax', 'Preco_MinMax',
    'Marca_Cod', 'Material_Cod', 'Temporada_Cod', 'Qtd_Vendidos_Cod',
    'Marca_Freq', 'Material_Freq'
]
colunas_numericas = [c for c in colunas_numericas if c in plot_df.columns]

corr = plot_df[colunas_numericas].corr(numeric_only=True)
fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    aspect='auto',
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    title='Mapa de Calor das Correlacoes'
)
fig_heatmap.update_layout(xaxis_title='Variaveis', yaxis_title='Variaveis')

# 4) Barra - top 10 marcas
marca_counts = plot_df['Marca'].value_counts().head(10).reset_index()
marca_counts.columns = ['Marca', 'Quantidade']
fig_bar = px.bar(
    marca_counts,
    x='Marca',
    y='Quantidade',
    title='Top 10 Marcas com Mais Produtos',
    labels={'Marca': 'Marca', 'Quantidade': 'Quantidade de Produtos'}
)

# 5) Pizza - distribuicao por temporada
if 'Temporada' in plot_df.columns:
    temp_counts = plot_df['Temporada'].value_counts().reset_index()
    temp_counts.columns = ['Temporada', 'Quantidade']
else:
    temp_counts = pd.DataFrame({'Temporada': ['Sem dado'], 'Quantidade': [1]})

fig_pie = px.pie(
    temp_counts,
    names='Temporada',
    values='Quantidade',
    title='Distribuicao de Produtos por Temporada'
)

# 6) Densidade - nota
x = plot_df['Nota'].dropna().values
hist, edges = np.histogram(x, bins=25, density=True)
centers = (edges[:-1] + edges[1:]) / 2
fig_density = go.Figure()
fig_density.add_trace(
    go.Scatter(
        x=centers,
        y=hist,
        mode='lines',
        fill='tozeroy',
        name='Densidade aproximada'
    )
)
fig_density.update_layout(
    title='Grafico de Densidade da Nota',
    xaxis_title='Nota',
    yaxis_title='Densidade'
)

# 7) Regressao - preco x quantidade vendida codificada
reg_df = plot_df[['Preco', 'Qtd_Vendidos_Cod']].dropna().copy()
coef = np.polyfit(reg_df['Preco'], reg_df['Qtd_Vendidos_Cod'], 1)
linha_x = np.linspace(reg_df['Preco'].min(), reg_df['Preco'].max(), 100)
linha_y = coef[0] * linha_x + coef[1]

fig_reg = go.Figure()
fig_reg.add_trace(
    go.Scatter(
        x=reg_df['Preco'],
        y=reg_df['Qtd_Vendidos_Cod'],
        mode='markers',
        name='Observacoes'
    )
)
fig_reg.add_trace(
    go.Scatter(
        x=linha_x,
        y=linha_y,
        mode='lines',
        name='Linha de regressao'
    )
)
fig_reg.update_layout(
    title='Grafico de Regressao: Preco x Quantidade Vendida',
    xaxis_title='Preco',
    yaxis_title='Qtd_Vendidos_Cod'
)

# =========================
# Aplicacao Dash
# =========================
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F7F7F7', 'padding': '20px'},
    children=[
        html.H1('Dashboard - Analise de Ecommerce', style={'textAlign': 'center'}),
        html.P(
            'Aplicacao Dash para visualizacao dos graficos do projeto de analise estatistica.',
            style={'textAlign': 'center', 'maxWidth': '900px', 'margin': '0 auto 24px auto'}
        ),

        html.Div([
            html.Div([dcc.Graph(figure=fig_hist)], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_scatter)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        html.Div([
            html.Div([dcc.Graph(figure=fig_heatmap)], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_bar)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        html.Div([
            html.Div([dcc.Graph(figure=fig_pie)], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_density)], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        html.Div([
            dcc.Graph(figure=fig_reg)
        ], style={'marginTop': '10px'}),
    ]
)

if __name__ == '__main__':
    app.run(debug=True)
