import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine, text
import locale

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ----------------------------------------------------------
st.set_page_config(
    page_title="Análise Histórica e Modelo Preditivo",
    page_icon="🎯",
    layout="wide"
)

try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except:
    pass

# ----------------------------------------------------------
# BANCO DE DADOS
# ----------------------------------------------------------
DB_USER = "data_iesb"
DB_PASSWORD = "iesb"
DB_HOST = "bigdata.dataiesb.com"
DB_PORT = "5432"
DB_NAME = "iesb"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def get_connection():
    try:
        return create_engine(DATABASE_URL, pool_pre_ping=True)
    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        return None

# ----------------------------------------------------------
# TREINO DO MODELO NACIONAL
# ----------------------------------------------------------
@st.cache_resource
def train_national_model(_engine):
    query = text("""
        WITH pop AS (
            SELECT "CO_MUNICIPIO",
                   SUM(CASE WHEN ("IDADE")::int BETWEEN 0 AND 17 THEN ("TOTAL") ELSE 0 END)
                   AS pop_criancas_adole
            FROM "Censo_20222_Populacao_Idade_Sexo"
            WHERE "ANO"::int = 2022
            GROUP BY "CO_MUNICIPIO"
        ), latest_esc AS (
            SELECT "CO_MUNICIPIO", qtd_escolas FROM (
                SELECT "CO_MUNICIPIO",
                       COUNT(DISTINCT "CO_ENTIDADE") AS qtd_escolas,
                       ROW_NUMBER() OVER (PARTITION BY "CO_MUNICIPIO" ORDER BY "NU_ANO_CENSO" DESC) AS rn
                FROM "educacao_basica"
                GROUP BY "CO_MUNICIPIO", "NU_ANO_CENSO"
            ) sub WHERE rn = 1
        ), latest_pib AS (
            SELECT "codigo_municipio_dv", vl_pib_per_capta FROM (
                SELECT "codigo_municipio_dv",
                       "vl_pib_per_capta",
                       ROW_NUMBER() OVER (PARTITION BY "codigo_municipio_dv" ORDER BY "ano_pib" DESC) AS rn
                FROM "pib_municipios"
            ) sub WHERE rn = 1
        )
        SELECT p."CO_MUNICIPIO", p.pop_criancas_adole, e.qtd_escolas, b.vl_pib_per_capta
        FROM pop p
        INNER JOIN latest_esc e ON p."CO_MUNICIPIO"::bigint = e."CO_MUNICIPIO"::bigint
        INNER JOIN latest_pib b ON p."CO_MUNICIPIO"::bigint = b."codigo_municipio_dv"::bigint
        WHERE p.pop_criancas_adole > 0
          AND e.qtd_escolas IS NOT NULL
          AND b.vl_pib_per_capta IS NOT NULL;
    """)

    df = pd.read_sql(query, _engine)
    if len(df) < 10:
        return None, None, None, None

    df['pop_criancas_adole_log'] = np.log1p(df['pop_criancas_adole'])

    X_raw = df[['pop_criancas_adole_log', 'vl_pib_per_capta']]
    y = df['qtd_escolas']

    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    model = LinearRegression().fit(X_scaled, y)
    preds = model.predict(X_scaled)

    stats = {
        'r2': r2_score(y, preds),
        'mae': mean_absolute_error(y, preds),
        'rmse': np.sqrt(mean_squared_error(y, preds)),
        'n_samples': len(df)
    }

    return model, stats, scaler, df

# ----------------------------------------------------------
# FUNÇÕES DE CONSULTA
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_municipios(_engine):
    return pd.read_sql(
        "SELECT nome_municipio, codigo_municipio_dv, latitude, longitude FROM municipio ORDER BY nome_municipio;",
        _engine
    )

@st.cache_data(ttl=3600)
def get_historico(_engine, cod):
    query = text("""
        WITH pib AS (
            SELECT "ano_pib" AS ano, "vl_pib_per_capta"
            FROM "pib_municipios"
            WHERE "codigo_municipio_dv" = :cod
        ), esc AS (
            SELECT "NU_ANO_CENSO" AS ano,
                   COUNT(DISTINCT "CO_ENTIDADE") AS qtd_escolas
            FROM "educacao_basica"
            WHERE "CO_MUNICIPIO"::bigint = :cod
            GROUP BY "NU_ANO_CENSO"
        )
        SELECT COALESCE(pib.ano, esc.ano) AS ano,
               pib.vl_pib_per_capta,
               esc.qtd_escolas
        FROM pib
        FULL OUTER JOIN esc ON pib.ano = esc.ano
        ORDER BY ano;
    """)
    return pd.read_sql(query, _engine, params={'cod': cod})

@st.cache_data(ttl=3600)
def get_latest_data_for_model(_engine, cod):
    queries = {
        'pop': text("""
            SELECT "ANO", SUM(CASE WHEN ("IDADE")::int BETWEEN 0 AND 17 THEN ("TOTAL") ELSE 0 END)
            FROM "Censo_20222_Populacao_Idade_Sexo"
            WHERE "CO_MUNICIPIO"::bigint = :cod
            GROUP BY "ANO"
            ORDER BY "ANO" DESC LIMIT 1
        """),
        'esc': text("""
            SELECT "NU_ANO_CENSO", COUNT(DISTINCT "CO_ENTIDADE")
            FROM "educacao_basica"
            WHERE "CO_MUNICIPIO"::bigint = :cod
            GROUP BY "NU_ANO_CENSO"
            ORDER BY "NU_ANO_CENSO" DESC LIMIT 1
        """),
        'pib': text("""
            SELECT "ano_pib", "vl_pib_per_capta"
            FROM "pib_municipios"
            WHERE "codigo_municipio_dv" = :cod
            ORDER BY "ano_pib" DESC LIMIT 1
        """)
    }

    results = {}
    with _engine.connect() as conn:
        for k, q in queries.items():
            r = conn.execute(q, {'cod': cod}).fetchone()
            results[k] = {'ano': r[0], 'valor': r[1]} if r else {'ano': 'N/A', 'valor': np.nan}

    return results

# ----------------------------------------------------------
# INTERFACE PRINCIPAL
# ----------------------------------------------------------
st.title("Análise por Município e Modelo Preditivo 🎯")

engine = get_connection()

if not engine:
    st.error("Erro ao conectar ao banco.")
    st.stop()

model, stats, scaler, df_train = train_national_model(engine)

df_mun = load_municipios(engine)
municipio_selecionado = st.sidebar.selectbox("Selecione um Município", df_mun['nome_municipio'])

if stats:
    with st.expander("Estatísticas do Modelo Preditivo Nacional"):
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{stats['r2']:.2%}")
        col2.metric("MAE", f"{stats['mae']:.2f}")
        col3.metric("RMSE", f"{stats['rmse']:.2f}")

# ----------------------------------------------------------
# ANÁLISE DO MUNICÍPIO
# ----------------------------------------------------------
info = df_mun[df_mun['nome_municipio'] == municipio_selecionado].iloc[0]
cod = info['codigo_municipio_dv']

st.header(f"Município: {municipio_selecionado}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Localização")
    st.map(pd.DataFrame({'lat': [info['latitude']], 'lon': [info['longitude']]}), zoom=8)

    st.subheader("Histórico")
    df_hist = get_historico(engine, cod)
    st.dataframe(df_hist)

with col2:
    st.subheader("Distribuição Nacional")

    df_plot = df_train.copy()
    df_plot['selected'] = df_plot['CO_MUNICIPIO'] == cod

    fig = px.scatter(
        df_plot,
        x='qtd_escolas',
        y='pop_criancas_adole_log',
        color='selected',
        hover_data=['vl_pib_per_capta']
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# PREDIÇÃO
# ----------------------------------------------------------
st.subheader("Predição pelo Modelo Nacional")

latest = get_latest_data_for_model(engine, cod)

pop_valor = latest['pop']['valor']
pib_valor = latest['pib']['valor']
real = latest['esc']['valor']

if pd.isna(pop_valor) or pd.isna(pib_valor) or pd.isna(real):
    st.warning("Dados insuficientes para prever.")
else:
    df_input = pd.DataFrame(
        [[pop_valor, pib_valor]],
        columns=['pop_criancas_adole', 'vl_pib_per_capta']
    )

    df_input['pop_criancas_adole_log'] = np.log1p(df_input['pop_criancas_adole'])
    X_proc = df_input[['pop_criancas_adole_log', 'vl_pib_per_capta']]

    X_scaled = scaler.transform(X_proc)
    esperado = model.predict(X_scaled)[0]
    diff = real - esperado

    colA, colB = st.columns(2)

    with colA:
        st.metric("Esperado", f"{esperado:.0f}")
        st.metric("Diferença", f"{diff:+.0f}")

    with colB:
        fig2 = px.bar(
            x=['Real', 'Esperado'],
            y=[real, esperado],
            text=[real, f"{esperado:.0f}"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    if diff >= 0:
        st.success(f"{municipio_selecionado} possui {diff:.0f} escolas **a mais** do que o esperado.")
    else:
        st.error(f"{municipio_selecionado} possui {abs(diff):.0f} escolas **a menos** do que o esperado.")
