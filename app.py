"""
Demo: classificação de sementes de trigo (Kama, Rosa, Canadian).

Rode com:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).parent
CORES = {'Kama': '#4C72B0', 'Rosa': '#55A868', 'Canadian': '#C44E52'}
ROTULOS = {
    'area': ('Área', 'mm²'),
    'perimetro': ('Perímetro', 'mm'),
    'compacidade': ('Compacidade', '4π·A/P²'),
    'comprimento_nucleo': ('Comprimento do núcleo', 'mm'),
    'largura_nucleo': ('Largura do núcleo', 'mm'),
    'coef_assimetria': ('Coeficiente de assimetria', ''),
    'comprimento_sulco': ('Comprimento do sulco', 'mm'),
}


@st.cache_resource
def carregar_modelo():
    return joblib.load(RAIZ / 'models' / 'modelo_sementes.joblib')


@st.cache_data
def carregar_dados():
    colunas = list(ROTULOS) + ['variedade']
    df = pd.read_csv(RAIZ / 'data' / 'seeds_dataset.txt', sep=r'\s+', header=None, names=colunas)
    df['variedade_nome'] = df['variedade'].map({1: 'Kama', 2: 'Rosa', 3: 'Canadian'})
    return df


st.set_page_config(page_title='Classificador de Sementes de Trigo', page_icon='🌾', layout='wide')

pacote = carregar_modelo()
modelo, features, classes = pacote['pipeline'], pacote['features'], pacote['classes']
df = carregar_dados()

st.title('🌾 Classificador de Sementes de Trigo')
st.markdown(
    'Informe as medidas físicas de um grão e o modelo indica a variedade: **Kama**, **Rosa** ou **Canadian**. '
    'Modelo: SVM com kernel linear, treinado no '
    '[Seeds Dataset (UCI)](https://archive.ics.uci.edu/dataset/236/seeds) · F1 de 0,96 na validação cruzada.'
)

# Exemplo: preenche os campos com um grão real do dataset
st.sidebar.header('Carregar um grão real')
st.sidebar.caption('Sorteia um grão do dataset para testar o modelo.')
for nome in CORES:
    if st.sidebar.button(f'Sortear um grão {nome}', use_container_width=True):
        amostra = df[df['variedade_nome'] == nome].sample(1).iloc[0]
        for f in features:
            st.session_state[f] = float(amostra[f])
        st.session_state['real'] = nome
if st.sidebar.button('Voltar para a mediana', use_container_width=True):
    for f in features:
        st.session_state[f] = float(df[f].median())
    st.session_state.pop('real', None)

st.sidebar.divider()
st.sidebar.markdown(
    'Projeto de [Tamires Ferreira](https://github.com/tamiresvferreira) · '
    '[código no GitHub](https://github.com/tamiresvferreira/classificacao-sementes-trigo)'
)

col_entrada, col_resultado = st.columns([1, 1.2], gap='large')

with col_entrada:
    st.subheader('Medidas do grão')
    valores = {}
    for f in features:
        nome, unidade = ROTULOS[f]
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        margem = (hi - lo) * 0.1
        st.session_state.setdefault(f, med)
        valores[f] = st.number_input(
            f'{nome}' + (f' ({unidade})' if unidade else ''),
            min_value=round(lo - margem, 4), max_value=round(hi + margem, 4),
            step=0.01 if f != 'compacidade' else 0.001,
            format='%.4f' if f == 'compacidade' else '%.3f',
            key=f,
            help=f'No dataset: de {lo:.3f} a {hi:.3f}',
        )

entrada = np.array([[valores[f] for f in features]])
proba = modelo.predict_proba(entrada)[0]
ordem_classes = [classes[c] for c in modelo.classes_]
previsto = ordem_classes[int(np.argmax(proba))]

with col_resultado:
    st.subheader('Resultado')
    st.markdown(
        f"<div style='padding:1rem 1.25rem;border-radius:0.75rem;border-left:6px solid {CORES[previsto]};"
        f"background:rgba(127,127,127,0.08)'>"
        f"<div style='font-size:0.9rem;opacity:0.8'>Variedade prevista</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{CORES[previsto]}'>{previsto}</div>"
        f"<div>Confiança: <b>{proba.max():.1%}</b></div></div>",
        unsafe_allow_html=True,
    )
    if 'real' in st.session_state:
        real = st.session_state['real']
        if real == previsto:
            st.success(f'Grão sorteado era **{real}**: o modelo acertou.')
        else:
            st.warning(f'Grão sorteado era **{real}**: o modelo errou. '
                       'Kama tem tamanho intermediário e às vezes se confunde com as vizinhas.')

    st.markdown('**Probabilidade por variedade**')
    st.bar_chart(
        pd.DataFrame({'Probabilidade': proba}, index=ordem_classes),
        horizontal=True, color='#2ca02c', height=180,
    )
    if proba.max() < 0.6:
        st.info('Confiança baixa: as medidas ficam na fronteira entre variedades.')

    st.markdown('**Onde o grão fica em relação ao dataset**')
    eixo_x = st.selectbox('Eixo X', features, index=features.index('area'),
                          format_func=lambda f: ROTULOS[f][0])
    eixo_y = st.selectbox('Eixo Y', features, index=features.index('comprimento_sulco'),
                          format_func=lambda f: ROTULOS[f][0])
    pontos = df[[eixo_x, eixo_y, 'variedade_nome']].rename(columns={'variedade_nome': 'Variedade'})
    novo = pd.DataFrame({eixo_x: [valores[eixo_x]], eixo_y: [valores[eixo_y]]})
    titulo_x, titulo_y = ROTULOS[eixo_x][0], ROTULOS[eixo_y][0]
    base = alt.Chart(pontos).mark_circle(size=55, opacity=0.55).encode(
        x=alt.X(eixo_x, title=titulo_x, scale=alt.Scale(zero=False)),
        y=alt.Y(eixo_y, title=titulo_y, scale=alt.Scale(zero=False)),
        color=alt.Color('Variedade', scale=alt.Scale(domain=list(CORES), range=list(CORES.values()))),
        tooltip=['Variedade', eixo_x, eixo_y],
    )
    seu_grao = alt.Chart(novo).mark_point(
        shape='diamond', size=260, filled=True, color='black', stroke='white', strokeWidth=1.5
    ).encode(x=eixo_x, y=eixo_y)
    st.altair_chart((base + seu_grao).properties(height=320), use_container_width=True)
    st.caption('◆ preto = o grão informado')

with st.expander('Como o modelo funciona'):
    st.markdown(
        f"""
- **Pipeline:** `StandardScaler` → `SVC(kernel='linear', C={pacote['params']['C']})`. A padronização faz parte
  do modelo salvo, então as medidas entram na escala original.
- **Avaliação:** validação cruzada estratificada repetida (5 folds × 10) com F1 macro de 0,964 ± 0,029 e
  88,9% de acurácia no conjunto de teste.
- **Retreino:** depois da avaliação, o modelo foi retreinado com os 210 grãos.
- **Limite conhecido:** 4 grãos do teste foram errados por todos os modelos testados. As medidas físicas
  sozinhas não bastam para eles.

Análise completa no [notebook](https://github.com/tamiresvferreira/classificacao-sementes-trigo/blob/main/notebooks/classificacao_sementes.ipynb).
"""
    )
