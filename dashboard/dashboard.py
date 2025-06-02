from enum import unique

import streamlit as st
import plotly.express as px
import os
import sys
import streamlit as st
from twisted.conch.scripts.tkconch import options

# Obtém o caminho do diretório raiz do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# Adiciona ao sys.path
sys.path.append(project_root)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.funcoes import *

st.set_page_config(layout="wide")
supabase = Sup_Cliente()

# Adicionando CSS via st.markdown
st.markdown("""
    <style>
        /* Banner fixo no topo */
        .top-banner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 160px;
            background-color: white;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Controle interno da imagem */
        .banner-left {
            display: flex;
            align-items: center;
            margin-top:64px;
            margin-left:40px;
            
        }

        /* Adiciona espaçamento ao conteúdo da página */
        .main > div:first-child {
            padding-top: 90px;
        }
    </style>
""", unsafe_allow_html=True)

logo_img = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(logo_img, "..", "assets/imagens/logo_sf.png"))
logo_base64 = image_to_base64(logo)

st.markdown(f"""
    <div class="top-banner">
        <div class="banner-left">
            <img src="data:image/png;base64,{logo_base64}" width="250"/>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<br/><br/>',unsafe_allow_html=True)
st.title("📊 Dashboard Automatizado Salvador em Dados - Beta")

abas = st.tabs(["🍉 Alimentação", "🏠 Habitação", "🚌 Transporte", "📈 ICV"])

# ============================= ABA ALIMENTAÇÃO =============================

with abas[0]:

    col_ano, col_mes = st.columns([2,2])
    col1, col2, col3 = st.columns([2, 1, 1])
    col4, col5 = st.columns([2,2])
    col6, col7 = st.columns([2,3])

    with col_ano:
        ano_selecionado = st.selectbox("Selecione o ano", options=carregar_select_box("cestas", coluna="ano"))

    with col_mes:
        mes_selecionado_alim = st.selectbox("Selecione o Mês", options=carregar_select_box("cestas", coluna="mes"))

    with col1:
        st.subheader("📊 Evolução do Custo em (R$) da Cesta Básica Salvador")
        df_evolucao = calcular_custo_cesta_evolucao()
        custo_cesta, primeiro_custo_cesta, df_cesta = calcular_custo_cesta_basica(mes_selecionado_alim,retornar_df=True)
        ultima = df_cesta.iloc[-1]
        horas = ultima["horas_trabalho"]
        salario = ultima["salario_estimado"]
        fig = px.line(df_evolucao, x="data_formatada", y="valor_cesta", markers=True)
        fig.update_layout(
            xaxis_title="Mês/Ano",
            yaxis_title="Valor da Cesta (R$)",
            title="Evolução do Valor da Cesta Básica",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"📌 Indicadores de Custo em {mes_selecionado_alim}")
        st.metric(label="🛒 Custo da Cesta Básica", value=f"R$ {custo_cesta:.2f}")
        st.metric(label="💰 Salário Mínimo Estimado", value=f"R$ {salario:.2f}")
        st.metric(label="⏱️ Horas Trabalhadas Necessárias", value=f"{horas:.2f}")
        st.metric(label="📉 Variação Acumulada do Ano",
                  value=f"{((custo_cesta-primeiro_custo_cesta)/primeiro_custo_cesta) * 100 :.2f}%")

    with col3:
        st.subheader("💸 Comprometimento do Salário")
        grafico_comprometimento_cesta_basica(custo_cesta, tipo='cesta')


    with col4:
        # ========================= GRÁFICO QUE ANALISA VARIAÇÃO DE CADA CATEGORIA DA CESTA BÁSICA =============================
        st.subheader("🔍  Evolução do Custo por Categoria da Cesta Básica")
        df_cesta_categoria = custo_cesta_categoria()

    with col5:
        st.subheader("🔍  Comprometimento Salário Mínimo por Categoria")
        variacao_categoria_por_data(df_cesta_categoria, mes_selecionado_alim, ano_selecionado, tipo=1)

    with col6:
       variacao_categoria_por_data(df_cesta_categoria, mes_selecionado_alim, ano_selecionado, tipo=2)


    with col7:
        st.subheader("⚙️ Benchmark Mensal por Local de Pesquisa")
        grafico_benchmark_alimentacao()


    # ============================= GRÁFICO COM A VARIAÇÃO ENTRE O MÊS ANTERIOR E O MÊS ATUAL NUMERICAMENTE =============================

    # Garantir que está ordenado corretamente
    st.subheader(f"🔍  Variação percentual itens da Cesta Básica - {mes_selecionado_alim}")
    # Ordena por categoria e data
    df = df_cesta_categoria.sort_values(["categoria", "data"])


    calcular_variacao_categorias_cesta(df, mes_selecionado_alim, ano_selecionado)
    st.markdown("<br>", unsafe_allow_html=True)

    # ============================= CÁLCULO DA PREVISÃO DO CUSTO CESTA USANDO ML PARA 30 DIAS =============================

    st.subheader("🍎 Previsão do Custo da Alimentação com Conjunto de Testes - Ensaio de Tendência")
    calcular_previsao_30_dias()

st.markdown("---")


# ============================= ABA HABITAÇÃO ===========================================================
with abas[1]:
    col_ano, col_mes = st.columns([2,2])
    col1, col2, col3 = st.columns([2, 1, 1])
    with col_ano:
        ano_selecionado_hab = st.selectbox("Selecione o Ano", options=carregar_select_box("custo_habitacao", coluna="ano"))

    with col_mes:
        mes_selecionado_hab = st.selectbox("Selecione o Mês", options=carregar_select_box("custo_habitacao", coluna="mes"))
    with col1:
        st.subheader("📊 Evolução do Custo de Habitação por Tipo de Imóvel")
        custo_habitacao, df_habitacao = calcular_custo_mensal_habitacao_valor(mes_selecionado_hab, ano_selecionado_hab, retornar_df=True)
        casa, kitnet, apartamento = exibir_resultados_habitacao(mes_selecionado_hab, ano_selecionado_hab) #retorno os valores calculados para a coluna 2


    with col2:
        if custo_habitacao is not None:
            st.subheader(f"📌 Indicadores de Custo Habitacional {mes_selecionado_hab}")
            st.metric(label="🏠 Custo Total Médio Ponderado com IPTU e Condomínio", value=f"R$ {custo_habitacao:.2f}")
            st.metric(label="🏙 Custo Médio Ponderado Apartamentos sem Taxas", value=f"R$ {apartamento:.2f}")
            st.metric(label="🏘 Custo Médio Ponderado Casas sem Taxas", value=f"R$ {casa:.2f}")
            st.metric(label="🏚 Custo Médio Ponderado Quartos sem Taxas", value=f"R$ {kitnet:.2f}")

    with col3:
        st.subheader("💸 Comprometimento do Salário")
        grafico_comprometimento_cesta_basica(custo_habitacao, tipo='habitacao')

# ========================== ÍNDICE DE HABITAÇÃO CRIADO PARA VALIDAR ================================

    #Rotina para fazer o benchmark da habitação usando o IPCA habitação
    st.subheader("📊 Índice Comparativo IPCA Habitação")
    custos, meses = obter_custos_habitacao()
    ipca_pct = [0.00,1.19]
    result = calcular_indice_base100_comparativo(custos,ipca_pct,meses)
    # Exibir gráficos comparativos
    st.line_chart(result.set_index('mes')[['indice_base100_hab', 'indice_base100_ipca']])

# ========================== TABELA COM OS CUSTOS HABITACIONAIS COM IMPOSTO E SEM ================================

    criar_tabela_custos_habitacao(df_habitacao, mes_selecionado_hab, ano_selecionado_hab)

# ========================== MAPA INTERATIVO ================================
    df_mapa = calcular_preco_medio_aluguel_por_bairro()

# ========================== ALUGUEIS POR BAIRRO ================================
    st.subheader("🏠 Custo Médio em (R$) de Aluguel por Bairro")
    bairro_selecionado = st.selectbox("Selecione o bairro", options=carregar_bairros(df_mapa))
    listar_preco_medio_por_bairro(bairro_selecionado, df_mapa)


# ======================= ANÁLISE POR TIPO DE IMÓVEL ========================
    st.markdown("## 📈 Análise Detalhada por Tipo de Imóvel")
    df_tipo_imovel = analise_tipo_imovel()


# ===================== COMPARATIVO DE TIPOS DE IMÓVEL ======================
    st.markdown("## 🔍 Comparativo Geral dos Custos de Habitação")
    comparativo_tipo_imovel(df_tipo_imovel)

# ============================= ABA TRANSPORTE =============================
with abas[2]:
    col_ano, col_mes = st.columns([2,2])
    col1, col2, col3 = st.columns([2, 1, 1])
    col4, col5 = st.columns([2,2])
    with col_ano:
        ano_selecionado_trans = st.selectbox("Selecione o Ano", options=carregar_select_box("custos_transporte", coluna="ano"),
                                         key="select_trans_ano")
    with col_mes:
        mes_selecionado_trans = st.selectbox("Selecione o Mês", options=carregar_select_box("custos_transporte", coluna="mes"),
                                             key="select_trans_mes")

    with col1:
        st.subheader("📊 Evolução do Custo (R$) de Transporte em Salvador")
        df_evolucao_transp = calcular_custos_transporte()
        mostrar_grafico_evolucao_transporte(df_evolucao_transp)

        with col2:
            df_dados_transp = carregar_dados("coletas_tarifas_transporte_publico")
            indicadores = calcular_indicadores_transporte(df_dados_transp)
            #st.write(indicadores)
            if df_dados_transp is not None:
                st.subheader(f"📌 Indicadores de Custos de Transporte {mes_selecionado_hab}")
                st.metric(label="🚌 Custo Tarifa Transporte Público", value=f"R$ {df_dados_transp["tarifa"].iloc[-1]:.2f}")
                st.metric(label="🚎 Total de Viagens Mensais Simulado", value=f" {indicadores["viagens_mensais"].iloc[0]}")
                st.metric(label="🚆 Trajeto Simulado Ponto A -> Ponto B", value=f"{indicadores["origem_bairro"].iloc[0]}"
                                                                               f" x {indicadores["destino_bairro"].iloc[0]}")
                st.metric(label="🛺 Custo Total-Baixa Renda usando modal ônibus",
                          value=f"R$ {indicadores["custo_total"].iloc[0]:.2f}")

        with col3:
            st.subheader("💸 Comprometimento do Salário")
            grafico_comprometimento_transporte(df_evolucao_transp)

        # ======================= SIMULE SEU CUSTO COM TRANSPORTE ========================
        st.title("🚚 Simule seu custo com Transporte")
        with col4:
            st.subheader("Simulador de Custo de Transporte")

            # Seletor de tipo de transporte
            tipo = st.selectbox("Tipo de transporte", [
                "Transporte Público", "Carro", "Moto", "App de Corrida", "Bicicleta"
            ])

            distancia_diaria_km = st.number_input("Distância total diária (ida e volta, em km)", min_value=0.0,
                                                  value=20.0)
            dias_por_mes = st.number_input("Dias por mês em que usa esse transporte", min_value=1, value=22)

            if tipo == "Transporte Público":
                tarifa = st.number_input("Tarifa por viagem (R$)", min_value=0.0, value=4.40)
                viagens_por_dia = st.selectbox("Número de viagens por dia", [1, 2, 3, 4], index=1)
                custo_mensal = tarifa * viagens_por_dia * dias_por_mes

            elif tipo in ["Carro", "Moto"]:
                preco_combustivel = st.number_input("Preço do combustível por litro (R$)", min_value=0.0, value=5.50)
                consumo = st.number_input("Consumo médio (km/l)", min_value=1.0, value=12.0)
                custo_mensal = (distancia_diaria_km * dias_por_mes / consumo) * preco_combustivel

            elif tipo == "App de Corrida":
                preco_por_km = st.number_input("Preço médio por km (R$)", min_value=0.0, value=2.00)
                corrida_minima = st.number_input("Valor mínimo por corrida (R$)", min_value=0.0, value=7.00)
                corridas_por_dia = st.selectbox("Número de corridas por dia", [1, 2, 3], index=1)
                custo_mensal = dias_por_mes * corridas_por_dia * max(
                    preco_por_km * (distancia_diaria_km / corridas_por_dia), corrida_minima)

            elif tipo == "Bicicleta":
                manutencao_mensal = st.number_input("Custo estimado com manutenção mensal (R$)", min_value=0.0,
                                                    value=10.00)
                custo_mensal = manutencao_mensal

            # Exibe o resultado
            st.subheader(f"**Custo mensal estimado com {tipo.lower()}: R$ {custo_mensal:.2f}**")

        with col5:
            grafico_comprometimento_cesta_basica(custo_mensal, "transporte")


rodape()