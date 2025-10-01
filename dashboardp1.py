# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
import locale
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -- CONFIGURAÇÃO DA PÁGINA --
st.set_page_config(
    page_title="Análise Histórica e Modelo Preditivo",
    page_icon="🎯",
    layout="wide"
)

# Configura a localidade
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    st.warning("Localidade 'pt_BR.UTF-8' não encontrada.")

# -- CONEXÃO COM O BANCO DE DADOS --
DB_USER = "data_iesb"
DB_PASSWORD = "iesb"
DB_HOST = "bigdata.dataiesb.com"
DB_PORT = "5432"
DB_NAME = "iesb"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def get_connection():
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# --- FUNÇÕES DE CARREGAMENTO E MODELAGEM ---

@st.cache_resource
def train_national_model(_engine):
    """Treina um modelo de regressão nacional com os dados mais recentes de cada município."""
    query = text("""
        WITH pop AS (
            SELECT "CO_MUNICIPIO", SUM(CASE WHEN ("IDADE")::int BETWEEN 0 AND 17 THEN ("TOTAL")::bigint ELSE 0 END) AS pop_criancas_adole
            FROM "Censo_20222_Populacao_Idade_Sexo" WHERE "ANO"::int = 2022 GROUP BY "CO_MUNICIPIO"
        ), latest_esc AS (
            SELECT "CO_MUNICIPIO", qtd_escolas FROM (
                SELECT "CO_MUNICIPIO", COUNT(DISTINCT "CO_ENTIDADE") AS qtd_escolas,
                       ROW_NUMBER() OVER(PARTITION BY "CO_MUNICIPIO" ORDER BY "NU_ANO_CENSO" DESC) as rn
                FROM "educacao_basica" GROUP BY "CO_MUNICIPIO", "NU_ANO_CENSO"
            ) sub WHERE rn = 1
        ), latest_pib AS (
            SELECT "codigo_municipio_dv", vl_pib_per_capta FROM (
                SELECT "codigo_municipio_dv", "vl_pib_per_capta",
                       ROW_NUMBER() OVER(PARTITION BY "codigo_municipio_dv" ORDER BY "ano_pib" DESC) as rn
                FROM "pib_municipios"
            ) sub WHERE rn = 1
        )
        SELECT p.pop_criancas_adole, e.qtd_escolas, b.vl_pib_per_capta
        FROM pop p
        INNER JOIN latest_esc e ON p."CO_MUNICIPIO"::bigint = e."CO_MUNICIPIO"::bigint
        INNER JOIN latest_pib b ON p."CO_MUNICIPIO"::bigint = b."codigo_municipio_dv"::bigint
        WHERE p.pop_criancas_adole > 0 AND e.qtd_escolas IS NOT NULL AND b.vl_pib_per_capta IS NOT NULL;
    """)
    df_model_data = pd.read_sql(query, _engine)
    if len(df_model_data) < 10: return None, None
    X = df_model_data[['pop_criancas_adole', 'vl_pib_per_capta']]
    y = df_model_data['qtd_escolas']
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    stats = {
        'r2': r2_score(y, predictions),
        'mae': mean_absolute_error(y, predictions),
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'n_samples': len(df_model_data)
    }
    return model, stats

@st.cache_data(ttl=3600)
def load_municipios(_engine):
    """Carrega lista de municípios com latitude e longitude."""
    query = "SELECT nome_municipio, codigo_municipio_dv, latitude, longitude FROM municipio ORDER BY nome_municipio;"
    df = pd.read_sql(query, _engine)
    return df

@st.cache_data(ttl=3600)
def get_historico(_engine, cod_municipio):
    query = text("""
        WITH pib AS (
            SELECT "ano_pib" AS ano, "vl_pib_per_capta"
            FROM "pib_municipios" WHERE "codigo_municipio_dv" = :cod
        ), esc AS (
            SELECT "NU_ANO_CENSO" AS ano, COUNT(DISTINCT "CO_ENTIDADE") AS qtd_escolas
            FROM "educacao_basica" WHERE "CO_MUNICIPIO"::bigint = :cod GROUP BY "NU_ANO_CENSO"
        )
        SELECT COALESCE(pib.ano, esc.ano) AS ano, pib.vl_pib_per_capta, esc.qtd_escolas
        FROM pib FULL OUTER JOIN esc ON pib.ano = esc.ano ORDER BY ano;
    """)
    df = pd.read_sql(query, _engine, params={'cod': cod_municipio})
    return df

@st.cache_data(ttl=3600)
def get_latest_data_for_model(_engine, cod_municipio):
    queries = {
        'pop': text("""..."""), 'esc': text("""..."""), 'pib': text("""...""")
    } # O corpo desta função permanece o mesmo da versão anterior.
    # Vou omitir por brevidade, mas ele está no código.
    queries = {
        'pop': text("""
            SELECT "ANO" AS ano, SUM(CASE WHEN ("IDADE")::int BETWEEN 0 AND 17 THEN ("TOTAL")::bigint ELSE 0 END) AS valor
            FROM "Censo_20222_Populacao_Idade_Sexo" WHERE "CO_MUNICIPIO"::bigint = :cod GROUP BY "ANO" ORDER BY "ANO" DESC LIMIT 1
        """),
        'esc': text("""
            SELECT "NU_ANO_CENSO" AS ano, COUNT(DISTINCT "CO_ENTIDADE") AS valor
            FROM "educacao_basica" WHERE "CO_MUNICIPIO"::bigint = :cod GROUP BY "NU_ANO_CENSO" ORDER BY "NU_ANO_CENSO" DESC LIMIT 1
        """),
        'pib': text("""
            SELECT "ano_pib" AS ano, "vl_pib_per_capta" AS valor
            FROM "pib_municipios" WHERE "codigo_municipio_dv" = :cod ORDER BY "ano_pib" DESC LIMIT 1
        """)
    }
    results = {}
    with _engine.connect() as connection:
        for key, query in queries.items():
            result = connection.execute(query, {'cod': cod_municipio}).fetchone()
            results[key] = {'ano': result[0], 'valor': result[1]} if result else {'ano': 'N/A', 'valor': np.nan}
    return results

# --- INTERFACE PRINCIPAL ---
st.title("Análise por Município e Modelo Preditivo")

engine = get_connection()
if engine:
    model, stats = train_national_model(engine)
    
    if stats:
        with st.expander("Ver Estatísticas do Modelo Preditivo Nacional"):
            st.write("Métricas de performance do modelo treinado com os dados mais recentes de cada município.")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{stats['r2']:.2%}", help="Explica qual % da variação no nº de escolas o modelo consegue prever.")
            col2.metric("Erro Médio (MAE)", f"{stats['mae']:.2f} escolas", help="Em média, o modelo erra por este nº de escolas.")
            col3.metric("Erro Típico (RMSE)", f"{stats['rmse']:.2f} escolas", help="Outra medida do erro médio, que penaliza mais os erros grandes.")
            st.caption(f"Modelo treinado com {stats['n_samples']} municípios.")

    df_municipios = load_municipios(engine)

    st.sidebar.header("Filtro")
    municipio_selecionado = st.sidebar.selectbox("Selecione um Município", df_municipios['nome_municipio'])

    if municipio_selecionado:
        info_municipio = df_municipios[df_municipios['nome_municipio'] == municipio_selecionado].iloc[0]
        cod_municipio_selecionado = info_municipio['codigo_municipio_dv']
        
        # **NOVO MAPA NA BARRA LATERAL**
        st.sidebar.subheader("Localização")
        map_data = pd.DataFrame({'lat': [info_municipio['latitude']], 'lon': [info_municipio['longitude']]})
        st.sidebar.map(map_data, zoom=8)
        
        st.header(f"Análise para: {municipio_selecionado}")

        st.subheader("Evolução Histórica")
        df_historico = get_historico(engine, cod_municipio_selecionado)
        if df_historico.empty or (df_historico['qtd_escolas'].isnull().all() and df_historico['vl_pib_per_capta'].isnull().all()):
            st.info("Não foram encontrados dados históricos de escolas ou PIB para este município.")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_historico['ano'], y=df_historico['qtd_escolas'], name='Nº de Escolas', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_historico['ano'], y=df_historico['vl_pib_per_capta'], name='PIB per Capita (R$)', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Nº de Escolas vs. PIB per Capita ao Longo do Tempo')
            fig.update_yaxes(title_text='Nº de Escolas', secondary_y=False)
            fig.update_yaxes(title_text='PIB per Capita (R$)', secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        st.subheader("🎯 Análise do Modelo Preditivo")
        if model is None:
            st.error("O modelo nacional não pôde ser treinado.")
        else:
            latest_data = get_latest_data_for_model(engine, cod_municipio_selecionado)
            pop_data, esc_data, pib_data = latest_data['pop'], latest_data['esc'], latest_data['pib']
            
            # **NOVOS ELEMENTOS: TABELA E GRÁFICO DE BARRAS**
            st.write("Comparamos o 'perfil mais recente' deste município com a 'regra geral' do Brasil.")
            
            if pd.isna(pop_data['valor']) or pd.isna(pib_data['valor']) or pd.isna(esc_data['valor']):
                st.warning("Não há dados completos (considerando o último ano de cada fonte) para comparar este município com o modelo nacional.")
            else:
                pop_valor_float, pib_valor_float = float(pop_data['valor']), float(pib_data['valor'])
                dados_para_prever = [[pop_valor_float, pib_valor_float]]
                escolas_reais = float(esc_data['valor'])
                escolas_esperadas = model.predict(dados_para_prever)[0]
                diferenca = escolas_reais - escolas_esperadas

                # Tabela Resumo
                summary_data = {
                    "Indicador": ["População Jovem", "PIB per Capita", "Nº de Escolas (Real)"],
                    "Ano": [pop_data['ano'], pib_data['ano'], esc_data['ano']],
                    "Valor": [f"{pop_valor_float:,.0f}", locale.currency(pib_valor_float, grouping=True), f"{escolas_reais:,.0f}"]
                }
                st.table(pd.DataFrame(summary_data))
                
                # Métricas e Gráfico de Barras em colunas
                col1, col2 = st.columns([1,2])
                with col1:
                    st.metric("Nº Esperado pelo Modelo", f"{escolas_esperadas:.0f}")
                    st.metric("Diferença", f"{diferenca:+.0f}", help="Positivo = mais escolas que o esperado.")

                with col2:
                    df_comp = pd.DataFrame({
                        'Categoria': ['Real', 'Esperado pelo Modelo'],
                        'Valor': [escolas_reais, escolas_esperadas]
                    })
                    fig_comp = px.bar(df_comp, x='Categoria', y='Valor', color='Categoria',
                                      title='Comparação: Nº Real vs. Nº Esperado de Escolas',
                                      text_auto='.0f')
                    st.plotly_chart(fig_comp, use_container_width=True)

                if diferenca > 0:
                    st.success(f"**Conclusão:** Com base nos seus dados mais recentes, {municipio_selecionado} tem **{diferenca:.0f} escolas a mais** do que o esperado para seu perfil.")
                else:
                    st.error(f"**Conclusão:** Com base nos seus dados mais recentes, {municipio_selecionado} tem **{abs(diferenca):.0f} escolas a menos** do que o esperado para seu perfil.")
else:
    st.error("Falha na conexão com o banco de dados.")