import streamlit as st
import pandas as pd
import logging
import plotly.express as px
import numpy as np
import locale
import base64
import calendar

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db.Sup_Cliente import Sup_Cliente
from io import BytesIO
from datetime import date
from babel.dates import format_date
from datetime import date

format_date(date.today(), format='long', locale='pt_BR')

supabase = Sup_Cliente()
salario_minimo = 1518.0

categorias_corretas = {
        "acucar": "Açúcar",
        "arroz": "Arroz",
        "banana": "Banana",
        "cafe": "Café",
        "carne": "Carne",
        "farinha": "Farinha",
        "batata": "Batata",
        "feijao": "Feijão",
        "leite": "Leite",
        "manteiga": "Manteiga",
        "oleo": "Óleo",
        "pao": "Pão",
        "tomate":"Tomate"

    }

quantidades_cesta = {
        "carne": 4.5, "leite": 6.0, "feijao": 4.5, "arroz": 3.6, "farinha": 3.0,
        "batata": 6.0, "tomate": 12.0, "pao": 6.0, "cafe": 300, "banana": 7.5,
        "acucar": 3.0, "oleo": 750, "manteiga": 750
    }

unidades_cesta = {
    "carne": "kg", "leite": "L", "feijao": "kg", "arroz": "kg", "farinha": "kg",
    "batata": "kg", "tomate": "kg", "pao": "kg", "cafe": "g", "banana": "kg",
    "acucar": "kg", "oleo": "ml", "manteiga": "g"
}

icones_png = {
    "arroz": "assets/icones/rice.png",
    "feijao": "assets/icones/coffee-beans.png",
    "carne": "assets/icones/meat.png",
    "leite": "assets/icones/milk.png",
    "manteiga": "assets/icones/butter.png",
    "banana": "assets/icones/banana.png",
    "tomate": "assets/icones/tomate.png",
    "farinha": "assets/icones/wheat-flour.png",
    "pao": "assets/icones/white-bread.png",
    "batata": "assets/icones/yam.png",
    "cafe": "assets/icones/tea.png",
    "oleo": "assets/icones/cooking-oil.png",
    "acucar": "assets/icones/sugar.png"

}
meses_map = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
    "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
    "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}

def rodape():
    st.subheader('Direitos Autorais')
    st.write('Todos os Direitos Reservados ©2025 - Prof. Matheus Brito de Oliveira')
    st.write('Todas as fontes são obtidas pelo próprio sistema.')
    st.write('Para qualquer esclarecimento/solicitação - entre em contato através do link: colocar o site')

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def imagem_para_base64(path_img):
    with open(path_img, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

def carregar_dados(tabela):
    response = supabase.buscar(tabela)
    return pd.DataFrame(response)


def calcular_ano(valores):
    anos = []
    for valor in valores:
        try:
            ano_num = int(float(valor))
            if 2025 <= ano_num <= 2030:
                anos.append(ano_num)
        except Exception as e:
            st.warning(f"Erro ao processar ano '{valor}': {e}")
    anos_ordenados = sorted(set(anos))
    return anos_ordenados

def calcular_mes(valores):
    meses = []
    for valor in valores:
        try:
            mes_num = int(float(valor))  # transforma 2.025 → 2
            if 1 <= mes_num <= 12:
                meses.append(calendar.month_name[mes_num])
        except Exception as e:
            st.warning(f"Erro ao processar '{valor}': {e}")

    meses_ordenados = sorted(set(meses), key=lambda m: list(calendar.month_name).index(m))
    return meses_ordenados


# ========================================= FUNÇÕES MÓDULO ALIMENTAÇÃO ===============================================

def carregar_select_box(modulo, coluna):
    df = carregar_dados(modulo)

    # Garantir que a coluna existe
    if coluna not in df.columns:
        st.error("Coluna não encontrada.")
        return []

    # Remove nulos e pega valores únicos
    valores = df[coluna].dropna().unique()

    if coluna == "mes":
        return calcular_mes(valores)

    elif coluna == "ano":
        return calcular_ano(valores)

    else:
        return sorted(valores.tolist())


def grafico_comprometimento_cesta_basica(custo, tipo='cesta'):
    salario_minimo = 1518.0
    comprometido = min(custo, salario_minimo)
    restante = max(salario_minimo - custo, 0)
    if tipo == 'cesta':
        labels = ['Cesta Básica', 'Demais Gastos']
    elif tipo == 'habitacao':
        labels = ['Habitação', 'Demais Gastos']
    else:
        labels = ['Transporte', 'Demais Gastos']

    valores = [comprometido, restante]
    cores = {
        'Cesta Básica': '#F4B400',
        'Habitação': '#1A2E4F',
        'Transporte': '#F4B400',
        'Demais Gastos': '#4C80D4'
    }

    fig = px.pie(
        names=labels,
        values=valores,
        hole=0.5,
        color=labels,
        color_discrete_map=cores
    )

    fig.update_traces(textinfo="percent+label")

    fig.update_layout(
        margin=dict(t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def custo_cesta_categoria():
    response = supabase.rpc_no_param("get_valores_por_categoria")
    df_cesta_categoria =  pd.DataFrame(response.data)
    df_cesta_categoria["data"] = pd.to_datetime(df_cesta_categoria["ano"].astype(str) + "-" +
                                                df_cesta_categoria["mes"].astype(str).str.zfill(2))

    fig = px.line(
        df_cesta_categoria,
        x="data",
        y="gasto_total",
        markers=True,
        color="categoria",
        title=""
    )
    st.plotly_chart(fig)
    return df_cesta_categoria

def calcular_custo_cesta_evolucao():
    df = carregar_dados("cestas")
    df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    df = df.sort_values("data")

    df["data_formatada"] = df["data"].dt.strftime("%m/%Y")

    return df[["data_formatada", "valor_cesta"]]

def calcular_custo_cesta_basica(mes_nome, retornar_df=False):
    df = carregar_dados("cestas")

    # Garantir que o mês é válido
    if mes_nome not in calendar.month_name:
        raise ValueError(f"Mês inválido: {mes_nome}")

    mes_num = list(calendar.month_name).index(mes_nome)

    # Processamento da data
    df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    df["ordem_custom"] = (df["data"] - df["data"].min()).dt.days
    df = df.sort_values("ordem_custom")
    df["data_formatada"] = df["data"].dt.strftime("%m/%Y")

    df = df.rename(columns={
        "horas_trabalho_necessarias": "horas_trabalho",
        "salario_minimo_necessario": "salario_estimado"
    })

    # Filtrar apenas registros do mês escolhido
    df_mes = df[df["mes"] == mes_num]

    if df_mes.empty:
        return (None, None, df) if retornar_df else None

    # Pega o registro mais recente desse mês
    linha_mais_recente = df_mes.sort_values(by=["ano", "mes"], ascending=[False, False]).iloc[0]
    linha_mais_antiga = df.sort_values(by=["ano", "mes"]).iloc[0]  # primeiro mes

    valor_mais_antigo = linha_mais_antiga["valor_cesta"]
    valor_cesta = linha_mais_recente["valor_cesta"]

    return (valor_cesta, valor_mais_antigo, df_mes) if retornar_df else valor_cesta


def variacao_categoria_por_data(df, mes_selecionado, ano_selecionado, tipo):
    mes = meses_map[mes_selecionado]
    df_filtrado = df[(df["mes"] == mes) & (df["ano"] == ano_selecionado)].copy()

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para o mês e ano selecionados.")
        return

    df_filtrado["percentual_salario"] = (df_filtrado["gasto_total"] / salario_minimo) * 100
    df_filtrado["categoria"] = df_filtrado["categoria"].map(categorias_corretas).fillna(df_filtrado["categoria"])

    if tipo == 1:
        # Gráfico de pizza
        fig = px.pie(
            df_filtrado,
            names="categoria",
            values="percentual_salario",
            title=f"Gastos por Categoria em {mes}/{ano_selecionado} (% do Salário Mínimo)",
            hole=0.5
        )
        fig.update_traces(
            textinfo='none',
            texttemplate='%{label}: %{value:.1f}%',
            textposition='outside',
            pull=[0.05] * len(df_filtrado),
            marker=dict(line=dict(color='white', width=2))
        )
        fig.update_layout(height=500, width=700, margin=dict(t=80, b=80, l=80, r=80))
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == 2:
        # Gráfico de barras horizontais
        fig = px.bar(
            df_filtrado,
            x="percentual_salario",
            y="categoria",
            orientation="h",
            text=df_filtrado["percentual_salario"].map(lambda x: f"{x:.1f}%"),
            labels={"percentual_salario": "% do Salário Mínimo", "categoria": "Categoria"},
            title=f"Gastos por Categoria como % do Salário Mínimo - {mes}/{ano_selecionado}"
        )
        fig.update_traces(marker_color='teal', textposition='outside')
        fig.update_layout(height=500, xaxis_tickformat=".1f", margin=dict(l=100, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Tipo de gráfico inválido. Use 1 para pizza e 2 para barras.")


def calcular_variacao_categorias_cesta(df, mes_selecionado, ano):
    df_ordenado = df.sort_values(["categoria", "data"]).copy()

    # 1. Pega os dois últimos registros por categoria


    # 2. Calcula variação mensal (últimos dois meses) #precisa ajustar por data
    df_ordenado["variacao_pct"] = df_ordenado.groupby("categoria")["gasto_total"].pct_change() * 100
    df_ordenado["variacao_fmt"] = df_ordenado["variacao_pct"].map(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-")

    # 3. Último mês

    resultado = df_ordenado[
        (df_ordenado["data"].dt.month == meses_map[mes_selecionado]) &
        (df_ordenado["data"].dt.year == ano)
        ]

    tabela = resultado[["categoria", "gasto_total", "variacao_fmt"]].rename(columns={
        "categoria": "Categoria",
        #"data": "Data",
        "gasto_total": "Gasto Total",
        "variacao_fmt": "Variação Mensal"
    })

    # 5. Variação acumulada até o mês selecionado
    df_ordenado["fator"] = df_ordenado.groupby("categoria")["gasto_total"].pct_change().fillna(0) + 1

    # Calcula produto acumulado dos fatores mês a mês
    df_ordenado["fator_acumulado"] = df_ordenado.groupby("categoria")["fator"].cumprod()

    # Pega apenas o mês selecionado
    df_selecionado = df_ordenado[
        (df_ordenado["data"].dt.month == meses_map[mes_selecionado]) &
        (df_ordenado["data"].dt.year == ano)
        ]

    # Calcula variação acumulada como (fator_acumulado - 1) * 100
    df_selecionado["variacao_acumulada"] = (df_selecionado["fator_acumulado"] - 1) * 100
    df_selecionado["variacao_acumulada_fmt"] = df_selecionado["variacao_acumulada"].map(
        lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-")

    tabela = tabela.merge(
        df_selecionado[["categoria", "variacao_acumulada_fmt"]].rename(columns={
            "categoria": "Categoria",
            "variacao_acumulada_fmt": "Variação Acumulada"
        }),
        on="Categoria",
        how="left"
    )
    tabela["Ícone"] = tabela["Categoria"].map(lambda cat: imagem_para_base64(icones_png[cat]))
    tabela["Quantidade Cesta"] = tabela["Categoria"].map(quantidades_cesta).fillna(tabela["Categoria"])
    tabela["Unidade"] = tabela["Categoria"].map(unidades_cesta).fillna("")

    # 6. Formata data
    #tabela["Data"] = tabela["Data"].dt.strftime("%b/%Y").str.title()

    # 7. Renderiza HTML com imagens
    html = """
    <table style="width:100%; border-collapse: collapse;">
    <thead>
    <tr style="text-align: center; border-bottom: 1px solid #ccc;">
    <th>Ícone</th><th>Categoria</th><th>Gasto Total</th><th>Quantidade na Cesta</th><th>Variação Mensal</th><th>Variação Acumulada</th>
    </tr>
    </thead>
    <tbody>
    """

    for _, row in tabela.iterrows():
        html += f"""
    <tr style="border-bottom: 1px solid #eee; text-align:center">
    <td><img src="{row['Ícone']}" width="30"></td>
    <td>{categorias_corretas[row['Categoria']]}</td>
    <td>R${row['Gasto Total']:.2f}</td>
   <td>{row['Quantidade Cesta']:.2f} {row['Unidade']}</td>
    <td>{row['Variação Mensal']}</td>
    <td>{row['Variação Acumulada']}</td>
    </tr>
    """

    html += "</tbody></table>"

    st.markdown(html, unsafe_allow_html=True)


def calcular_previsao_30_dias():
    df = carregar_dados("cestas")
    if len(df) < 3:
        st.warning("⚠️ São necessários pelo menos 3 meses de dados para fazer uma previsão com validação.")
    else:
        df = df.sort_values(by=["ano", "mes"]).reset_index(drop=True)

        df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str) + "-01")
        df["dias"] = (df["data"] - df["data"].min()).dt.days

        X = df[["dias"]]
        y = df["valor_cesta"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        # Modelo de regressão
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Avaliação
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        st.write(f"📈 MAE (Erro Médio Absoluto): R$ {mae:.2f}")
        st.write(f"📉 RMSE (Raiz do Erro Quadrático Médio): R$ {rmse:.2f}")
        # st.write(y_test - y_pred)

        # Previsões para os próximos 30 dias
        dias_futuros = np.array(range(df["dias"].max() + 1, df["dias"].max() + 31)).reshape(-1, 1)
        previsoes = modelo.predict(dias_futuros)

        # Criar datas futuras baseadas na última data do dataset
        datas_futuras = pd.date_range(start=df["data"].max() + pd.DateOffset(months=1), periods=30, freq="D")
        df_previsao = pd.DataFrame({"data": datas_futuras, "valor_cesta": previsoes})

        # Plot
        fig = px.line(df, x="data", y="valor_cesta", title="Previsão do Custo da Alimentação")
        fig.add_scatter(x=df.loc[X_test.index, "data"], y=y_pred, mode="markers",
                        name="Previsões de Teste", marker=dict(color="red"))
        fig.add_scatter(x=df_previsao["data"], y=df_previsao["valor_cesta"], mode="lines",
                        name="Previsão Futura", line=dict(dash="dash"))

        fig.update_xaxes(tickformat="%b/%Y", title_text="Data")
        fig.update_yaxes(title_text="Valor da Cesta (R$)")
        st.plotly_chart(fig, use_container_width=True)

def grafico_benchmark_alimentacao():

    # Dados simulados (substitua pelos seus dados reais)
    dados = {
        'Mês': ['Fev', 'Mar', 'Abr'],
        'DIEESE': [628.80, 633.58, 632.12],
        'SEI' : [592.16, 615.75, 613.07],
        'ICV-SSA' : [669.25, 622.65, 658.86]
    }

    # Criar DataFrame
    df = pd.DataFrame(dados)

    # Transformar para formato "longo"
    df_long = df.melt(id_vars='Mês', var_name='Local', value_name='Resultado')

    # Criar gráfico com plotly.express
    fig = px.line(
        df_long,
        x='Mês',
        y='Resultado',
        color='Local',
        markers=True,
        title=''
    )

    # Exibir no Streamlit
    st.plotly_chart(fig)


# =========================================== FUNÇÕES MÓDULO HABITAÇÃO =================================================
def calcular_custo_mensal_habitacao_valor(mes, ano,retornar_df=False):
    df = carregar_dados("custo_habitacao")
    if df is None or df.empty:
        return []


    df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    df_filtrado = df[(df["ano"] == ano) & (df["mes"] == meses_map[mes])]

    valor_mais_recente = df_filtrado.sort_values(by=["ano", "mes"]).iloc[-1]["media_aluguel_com_taxas"]
    return (valor_mais_recente, df_filtrado) if retornar_df else valor_mais_recente

def obter_custos_habitacao():
    df = carregar_dados("custo_habitacao")

    if df is None or df.empty:
        return [], []

    # Cria uma coluna de data para ordenar
    df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    df = df.sort_values("data")

    # Seleciona custos de abril e maio de 2025
    abril = df.loc[df["data"] == "2025-04-01", "media_aluguel_com_taxas"].values
    maio = df.loc[df["data"] == "2025-05-01", "media_aluguel_com_taxas"].values

    # Verificação de segurança
    abril_valor = abril[-1] if len(abril) > 0 else None
    maio_valor = maio[-1] if len(maio) > 0 else None

    custos = []
    custos.append(abril_valor)
    custos.append(maio_valor)

    meses = (
        df.sort_values("data")  # ordena pelas datas reais
        .drop_duplicates(subset="data")  # garante 1 entrada por mês
        ["data"]
        .dt.strftime("%b/%Y")
        .tolist()
    )

    return custos, meses



def obter_tipos_imoveis():
    tipos = supabase.buscar("tipos_imoveis")  # retorna lista de dicts, ex: [{"id": "...", "nome": "..."}]
    return {tipo["id"]: tipo["descricao"] for tipo in tipos}


def exibir_resultados_habitacao(mes, ano):
    df = consultar_custos_habitacao()

    if df is None or df.empty:
        st.warning("⚠️ Nenhum dado de custo de habitação encontrado.")
        return None

    tipo_imovel_map = obter_tipos_imoveis()
    df["tipo_imovel"] = df["tipo_imovel_id"].map(tipo_imovel_map)
    df = df.dropna(subset=["tipo_imovel"])
    df["mes_ano"] = df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2)

    # 1) Para o gráfico, use o DataFrame completo (sem filtro por mês/ano)
    df_longo = pd.melt(
        df,
        id_vars=["mes_ano", "tipo_imovel"],
        value_vars=["media_aluguel_sem_taxas", "media_aluguel_com_taxas"],
        var_name="tipo_valor",
        value_name="valor"
    )
    legenda_tipo_valor = {
        "media_aluguel_sem_taxas": "Aluguel Sem Taxas",
        "media_aluguel_com_taxas": "Aluguel Com Taxas"
    }
    df_longo["tipo_valor_legenda"] = df_longo["tipo_valor"].map(legenda_tipo_valor)
    df_longo["linha"] = df_longo["tipo_imovel"] + " - " + df_longo["tipo_valor_legenda"]

    fig = px.line(
        df_longo,
        x="mes_ano",
        y="valor",
        color="linha",
        title="Evolução do Custo de Habitação por Tipo de Imóvel",
        labels={"mes_ano": "Mês-Ano", "valor": "Valor (R$)", "linha": "Tipo de Imóvel"},
        markers=True
    )
    fig.update_layout(xaxis_tickformat="%m/%Y")
    st.plotly_chart(fig, use_container_width=True)

    # 2) DataFrame filtrado só para o mês e ano selecionados
    df_filtrado = df[(df["ano"] == ano) & (df["mes"] == meses_map[mes])]

    df_longo_filtrado = pd.melt(
        df_filtrado,
        id_vars=["mes_ano", "tipo_imovel"],
        value_vars=["media_aluguel_sem_taxas", "media_aluguel_com_taxas"],
        var_name="tipo_valor",
        value_name="valor"
    )

    df_sem_taxas = df_longo_filtrado[df_longo_filtrado["tipo_valor"] == "media_aluguel_sem_taxas"]

    if df_sem_taxas.empty:
        st.info("Nenhum valor de aluguel sem taxas encontrado para o mês/ano selecionado.")

    casa = df_sem_taxas.loc[df_sem_taxas["tipo_imovel"] == "Casas", "valor"].values[0] \
        if not df_sem_taxas[df_sem_taxas["tipo_imovel"] == "Casas"].empty else None

    kitnet = df_sem_taxas.loc[df_sem_taxas["tipo_imovel"] == "Aluguel de quartos", "valor"].values[0] \
        if not df_sem_taxas[df_sem_taxas["tipo_imovel"] == "Aluguel de quartos"].empty else None

    apartamento = df_sem_taxas.loc[df_sem_taxas["tipo_imovel"] == "Apartamentos", "valor"].values[0] \
        if not df_sem_taxas[df_sem_taxas["tipo_imovel"] == "Apartamentos"].empty else None

    return casa, kitnet, apartamento


def consultar_custos_habitacao():
    try:
        # Consulta os dados da tabela 'custo_habitacao'
        resultado = supabase.buscar("custo_habitacao")

        if not resultado:
            logging.warning("⚠️ Nenhum dado encontrado para custo de habitação.")
            return None

        # Caso o resultado seja uma lista, converta para DataFrame diretamente
        if isinstance(resultado, list):
            df = pd.DataFrame(resultado)
        else:
            df = pd.DataFrame(resultado.data)

        # Criando uma coluna 'mes_ano' para representar o mês e ano em formato 'mes-ano'
        df['mes_ano'] = df['mes'].astype(str) + '-' + df['ano'].astype(str)

        # Convertendo as colunas de valores para o tipo numérico (caso ainda não seja)
        df['media_aluguel_sem_taxas'] = pd.to_numeric(df['media_aluguel_sem_taxas'], errors='coerce')
        df['media_aluguel_com_taxas'] = pd.to_numeric(df['media_aluguel_com_taxas'], errors='coerce')

        # Preenchendo NaN com zero ou outro valor apropriado, caso necessário
        df.fillna(0, inplace=True)  # Substitui NaN por 0. Se preferir, pode usar df.mean() para média.

        # Exibe o DataFrame com os dados de custo de habitação
        return df

    except Exception as e:
        logging.error(f"❌ Erro ao consultar dados de custo de habitação: {e}")
        return None


def analise_tipo_imovel():
    df_habitacao = carregar_dados("custo_habitacao")
    df_habitacao["data"] = pd.to_datetime(
        df_habitacao["ano"].astype(str) + "-" + df_habitacao["mes"].astype(str).str.zfill(2))

    tipos_imovel = obter_tipos_imoveis()  # {uuid: nome}


    tipo_selecionado = st.selectbox("Escolha o tipo de imóvel", list(tipos_imovel.values()))

    nome_para_uuid = {v: k for k, v in tipos_imovel.items()}
    uuid_selecionado = nome_para_uuid[tipo_selecionado]

    df_filtrado = df_habitacao[df_habitacao["tipo_imovel_id"] == uuid_selecionado]

    if df_filtrado.empty:
        st.warning(f"⚠️ Não há dados para o tipo de imóvel selecionado: {tipo_selecionado}")
        return df_habitacao

    fig_tipo = px.line(
        df_filtrado,
        x="data",
        y="media_aluguel_sem_taxas",  # ou media_aluguel_com_taxas, conforme desejar
        title=f"Evolução do Aluguel Sem Taxas - {tipo_selecionado}",
        markers=True,
        labels={"media_aluguel_sem_taxas": "Valor (R$)", "data": "Data"}
    )
    fig_tipo.update_layout(xaxis_tickformat="%m/%Y")
    st.plotly_chart(fig_tipo, use_container_width=True)

    return df_habitacao

def comparativo_tipo_imovel(df_habitacao):
    # Supondo que df_habitacao tenha: data, tipo_imovel_id, media_aluguel_sem_taxas, media_aluguel_com_taxas

    # Primeiro, crie um dicionário para mapear UUID para nome do tipo (exemplo)
    tipos_imovel = obter_tipos_imoveis()  # {uuid: nome}
    df_habitacao["tipo_imovel_nome"] = df_habitacao["tipo_imovel_id"].map(tipos_imovel)

    # Agora faz o melt mantendo tipo_imovel_nome
    df_long = df_habitacao.melt(
        id_vars=["data", "tipo_imovel_nome"],
        value_vars=["media_aluguel_sem_taxas", "media_aluguel_com_taxas"],
        var_name="tipo_custo",
        value_name="valor"
    )

    nomes_legiveis = {
        "media_aluguel_sem_taxas": "Sem Taxas",
        "media_aluguel_com_taxas": "Com Taxas"
    }
    df_long["tipo_custo"] = df_long["tipo_custo"].map(nomes_legiveis)

    # Cria coluna combinada para legenda e agrupamento
    df_long["legenda"] = df_long["tipo_imovel_nome"] + " - " + df_long["tipo_custo"]

    tipos_escolhidos = st.multiselect(
        "Escolha os tipos para comparar",
        options=df_long["legenda"].unique(),
        default=df_long["legenda"].unique()
    )

    df_filtrado = df_long[df_long["legenda"].isin(tipos_escolhidos)]

    fig_comp = px.line(
        df_filtrado,
        x="data",
        y="valor",
        color="legenda",
        title="Comparação dos Custos de Habitação por Tipo de Imóvel e Custo",
        markers=True,
        labels={"data": "Data", "valor": "Valor (R$)", "legenda": "Tipo"}
    )
    fig_comp.update_xaxes(dtick="M1", tickformat="%b/%Y", tickangle=45)
    st.plotly_chart(fig_comp, use_container_width=True)


def criar_tabela_custos_habitacao(df, mes, ano):
    df_filtrado = df[
        (df['ano'] == ano) &
        (df['mes'] == meses_map[mes])
        ][['media_aluguel_sem_taxas', 'media_aluguel_com_taxas', 'data']]


    df_filtrado['descricao'] = ['Gasto Médio Mensal com Casa',
                                'Gasto Médio Mensal com Quarto',
                                'Gasto Médio Mensal com Apartamento',
                                'Gasto Médio Mensal Total, considerando (Casas, Aps, Quartos, etc...)'


                                ]

    df_filtrado["data"] = df_filtrado["data"].dt.strftime("%B/%Y").str.capitalize()
    mes_atual = df_filtrado['data'].iloc[0]

    df_filtrado = df_filtrado[['descricao', 'media_aluguel_sem_taxas','media_aluguel_com_taxas']]
    df_filtrado.columns = ['Descrição','Média Geral Alugéis com Impostos', 'Média Geral Aluguéis sem Impostos']
    st.subheader(f"📊 Custos Habitacionais com e sem Taxas - {mes_atual}")
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)

def carregar_dados_por_bairro():
    response = supabase.rpc_no_param("get_preco_medio_por_bairro")

    return pd.DataFrame(response.data)

def calcular_preco_medio_aluguel_por_bairro():
    st.subheader("🗺️ Mapa de Custo da Habitação por Região")
    df_mapa = carregar_dados_por_bairro()

    if df_mapa is None or df_mapa.empty:
        st.warning("Aviso: Nenhum dado para exibir no mapa.")
        st.stop()

    df_mapa["preco_formatado"] = df_mapa["preco_medio"].apply(
        lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    fig_mapa = px.scatter_mapbox(
        df_mapa,
        lat="latitude",
        lon="longitude",
        size="preco_medio",
        color="preco_medio",
        color_continuous_scale="YlOrRd",
        hover_name="bairro",
        hover_data={"preco_formatado": True, "preco_medio": False},
        title="Custo da Habitação por Localização",
        mapbox_style="carto-darkmatter"
    )

    st.plotly_chart(fig_mapa)
    return df_mapa

def carregar_bairros(df_mapa):

    # Garantir que a coluna existe
    if 'bairro' not in df_mapa.columns:
        st.error("Coluna não encontrada.")
        return []

    # Remove nulos e pega valores únicos
    valores = df_mapa["bairro"].dropna().unique()

    bairros = []
    for valor in valores:
        try:
            bairros.append(valor)
        except Exception as e:
            st.warning(f"Erro ao processar '{valor}': {e}")
    bairros_ordenados  = sorted(set(bairros))
    return bairros_ordenados

def listar_preco_medio_por_bairro(bairro_selecionado, df_mapa):
    # Filtra os dados para o bairro selecionado
    df_filtrado = df_mapa[df_mapa["bairro"] == bairro_selecionado]

    # Verifica se há dados
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para o bairro selecionado.")
        return

    # Converte para número, se necessário (remove "R$", "." e troca "," por ".")
    df_filtrado["preco_formatado"] = df_filtrado["preco_formatado"].replace(
        {"R\$": "", "\.": "", ",": "."}, regex=True
    ).astype(float)

    # Calcula a média do preço no bairro
    preco_medio_aluguel = df_filtrado["preco_formatado"].mean()

    # Exibe os resultados
    st.subheader(f"📍 Preço médio em **{bairro_selecionado}**")
    st.write(f"🔹 Aluguel médio: R$ {preco_medio_aluguel:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)



def calcular_indice_base100_comparativo(custos_habitacao, ipca_pct, meses=None):
    """
    Calcula o índice base 100 para dados monetários e variações de IPCA, comparando ambos.

    :param custos_habitacao: Lista de valores monetários (ex: custos mensais), ex: [2342.58, 2400.00, 2450.00]
    :param ipca_pct: Lista de variações mensais em porcentagem do IPCA, ex: [0.46, 0.38, 0.40]
    :param meses: (Opcional) Lista com os nomes dos meses correspondentes aos valores, ex: ['Mar', 'Abr', 'Mai']
    :return: DataFrame com os valores e índices base 100 para ambos
    """
    # Para o IPCA, vamos converter % acumulado para índice base 100
    # Exemplo: IPCA acumulado 0% no mês 1 → índice 100
    # IPCA acumulado 1.19% no mês 2 → índice 100*(1 + 1.19/100) = 101.19
    # e assim sucessivamente
    # Definir os meses se não forem fornecidos
    if meses is None:
        meses = [f"Mês {i + 1}" for i in range(len(custos_habitacao))]

    # Criar DataFrame com os valores reais (custo habitacional)
    df = pd.DataFrame({
        'mes': meses,
        'custo_total_medio': custos_habitacao
    })


    # Calcular o índice base 100 para custo habitacional
    df['indice_base100_hab'] = (df['custo_total_medio'] / df['custo_total_medio'].iloc[0]) * 100

    # Calcular o índice base 100 para IPCA
    ipca_indice = [100]
    for pct in ipca_pct:
        novo_valor = ipca_indice[-1] * (1 + pct / 100)
        ipca_indice.append(novo_valor)

    # Adicionar o índice de IPCA no DataFrame
    df['indice_base100_ipca'] = ipca_indice[1:]  # remove o primeiro índice (100)

    # Retornar o DataFrame
    return df


# =========================================== FUNÇÕES MÓDULO TRANSPORTE =================================================

def mostrar_grafico_evolucao_transporte(df_evolucao_transp):
    if not df_evolucao_transp.empty:
        fig = px.line(
            df_evolucao_transp,
            x="Mês",
            y="custo_total",
            color="nome_perfil",
            markers=True,
            title="Evolução do Custo de Transporte por Perfil"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum gráfico para exibir")


def calcular_custos_transporte():
    df = carregar_dados("custos_transporte")
    df_perfil = carregar_dados("perfis_usuarios_transporte")

    # Verificação inicial
    if df.empty or df_perfil.empty:
        st.warning("Algum dos DataFrames está vazio.")
        return pd.DataFrame()

    # Ajuste o nome da chave de junção conforme seu schema
    df = pd.merge(df, df_perfil, left_on="perfil_id", right_on="id", how="left")

    df["data"] = pd.to_datetime(df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
    df = df.sort_values("data")
    df["data_formatada"] = df["data"].dt.strftime("%m/%Y")
    df.rename(columns={
        "data_formatada": "Mês"
    }, inplace=True)

    return df[["Mês", "nome_perfil", "custo_total"]]


def grafico_comprometimento_transporte(df):


    if not df.empty:
        # Agrupa os custos totais por perfil
        df_agrupado = df.groupby("nome_perfil")["custo_total"].sum().reset_index()

        # Calcula percentual sobre o salário mínimo
        df_agrupado["percentage_sal_min"] = (df_agrupado["custo_total"] / salario_minimo) * 100
        total_percentual = round(df_agrupado["percentage_sal_min"].sum(), 2)

        if total_percentual > 100:
            st.warning(f"O total comprometido ultrapassa 100% do salário mínimo ({total_percentual:.2f}%).")

        # Cálculo correto dos "Demais gastos"
        percentual_restante = max(0.01, 100 - total_percentual) if total_percentual < 100 else 0
        df_demais = pd.DataFrame({
            "nome_perfil": ["Demais gastos"],
            "percentage_sal_min": [percentual_restante]
        })

        # Junta os dados e calcula valores em reais
        df_final = pd.concat([
            df_agrupado[["nome_perfil", "percentage_sal_min"]],
            df_demais
        ], ignore_index=True)
        df_final["valor_em_reais"] = (df_final["percentage_sal_min"] / 100) * salario_minimo

        # Gera gráfico
        fig = px.pie(
            df_final,
            names="nome_perfil",
            values="percentage_sal_min",
            hole=0.5,
            title="",
            custom_data=["valor_em_reais"]
        )

        fig.update_layout(
            height=500,
            legend=dict(
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Não há dados disponíveis para exibir o gráfico.")

def calcular_indicadores_transporte(df):
    df_trajeto = carregar_dados("trajetos_por_perfil")
    df_trajetos = carregar_dados("coletas_trajetos")
    df_perfis = carregar_dados("perfis_usuarios_transporte")

    if df.empty or df_trajeto.empty:
        st.warning("Dataframe vazio")
        return pd.DataFrame()

    df_trajeto = df_trajeto[['trajeto_id', 'perfil_id', 'viagens_mensais']]

    df_completo = pd.merge(df_trajeto, df_trajetos, left_on="trajeto_id", right_on="id", how="left")
    df_completo = pd.merge(df_completo, df_perfis, left_on="perfil_id", right_on="id", how="left",
                           suffixes=('_trajeto', '_perfil'))

    # Calcula custo total do transporte
    df_completo["custo_total"] = df_completo["viagens_mensais"] * df["tarifa"].iloc[-1]

    # Filtra apenas as colunas desejadas
    df_filtrado = df_completo[[
        "nome_perfil", "origem_bairro", "destino_bairro", "viagens_mensais", "custo_total"
    ]]

    return df_filtrado


# =========================================== FUNÇÕES MÓDULO ICV =================================================



def carregar_select_box_icv(tabela, coluna, tempo):
    df = carregar_dados(tabela)

    if df.empty:
        st.error("Não existem dados")
        return []

    #valores = df[coluna].dropna().unique()

    df = converter_data_mes_ano(df, coluna)

    valores_mes = df["mes"].dropna().unique()
    valores_ano = df["ano"].dropna().unique()

    if tempo == "mes":
        return calcular_mes(valores_mes)
    else:
        return calcular_ano(valores_ano)

def converter_data_mes_ano(df, coluna):

    # Converte a coluna para datetime
    df[coluna] = pd.to_datetime(df[coluna])

    # Extrai ano e mês
    df['ano'] = df[coluna].dt.year
    df['mes'] = df[coluna].dt.month

    return df

def mostrar_grafico_evolucao_icv():

    df = carregar_dados("icv")
    df = converter_data_mes_ano(df, "data_calculo")

    if not df.empty:
        df["Data"] = df["data_calculo"].dt.strftime("%m/%Y")
        fig = px.line(
        df,
        x="Data",

        y="valor_icv",
        markers=True,
        title="Evolução do ICV"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Nenhum gráfico para exibir")

    return df

def filtrar_df_ano_mes(df, ano, mes):

    df_filtrado = df[
        (df['ano'] == ano) &
        (df['mes'] == meses_map[mes])]

    return df_filtrado



def calcular_indicadores(df, ano, mes):
    df_filtrado = filtrar_df_ano_mes(df, ano, mes)

    icv = df_filtrado["valor_icv"]
    variacao = df_filtrado["variacao_percentual"]

    return icv, variacao


def obter_pesos_icv(ano, mes):
    # Monta a data no início do mês
    data_calculo = date(ano, meses_map[mes], 1).isoformat()

    # Busca o ICV desse mês
    icv = supabase.buscar("icv", filtros={"data_calculo": data_calculo})
    if not icv:
        st.warning(f"Nenhum ICV encontrado para {data_calculo}.")
        return None

    icv_id = icv[0]['id']


    # Busca os pesos usando o ID do ICV
    pesos = supabase.buscar("pesos_icv", filtros={"icv_id": icv_id})
    if not pesos:
        st.warning(f"Nenhum peso encontrado para o ICV ID {icv_id}.")
        return None

    return pesos[0]

def mostrar_grafico_composicao_icv(ano, mes):
    pesos = obter_pesos_icv(ano, mes)
    if not pesos:
        return

    categorias = ["Alimentação", "Habitação", "Transporte"]
    valores = [
        pesos["peso_alimentacao"],
        pesos["peso_habitacao"],
        pesos["peso_transporte"]
    ]

    df = {
        "Categoria": categorias,
        "Peso (%)": [v * 100 for v in valores]
    }


    fig = px.pie(df, names="Categoria",
                 values="Peso (%)")
    fig.update_layout(
        height=500,
        legend=dict(
            orientation="h",
            y=-0.3,
            x=0.5,
            xanchor="center"
        ))

    st.plotly_chart(fig)


def grafico_custo_vs_renda(df):
    """
    Espera um DataFrame com as colunas:
    - data_calculo (ex: '2025-04-01')
    - valor_icv (float): ICV em base 100
    - salario_minimo (float): valor nominal em R$
    - categoria (opcional): ex: localização ou faixa etária
    """

    df["data_calculo"] = pd.to_datetime(df["data_calculo"])

    # Supondo que o salário mínimo real esteja em df["salario_minimo"]
    if "salario_minimo" not in df.columns:
        df["salario_minimo"] = salario_minimo  # valor fixo para todos, se não tiver coluna

    # Converter o salário mínimo para base 100 (mesmo mês base do ICV)
    salario_base = df["salario_minimo"].iloc[0]
    df["salario_indexado"] = (df["salario_minimo"] / salario_base) * 100

    fig = px.scatter(
        df,
        x="salario_indexado",
        y="valor_icv",
        color="categoria" if "categoria" in df.columns else None,
        hover_name="data_calculo",
        trendline="ols"
    )

    fig.update_layout(
        xaxis_title="Salário Mínimo (base 100)",
        yaxis_title="ICV (base 100)"
    )

    st.plotly_chart(fig)



def grafico_icv_vs_indicador(df, mes, indicador="ipca"):
    """
    Compara ICV com um indicador externo (ipca, selic, ibov).
    """


    df["data_calculo"] = pd.to_datetime(df["data_calculo"])
    df["ipca"] = 0.16
    #df_merge = pd.merge(df, df_ipca, on="data_calculo", how="inner")
    fig = px.scatter(
        df,
        x=indicador,
        y="valor_icv",
        hover_name="data_calculo",
        trendline="ols",  # adiciona linha de tendência
        title=f"ICV vs {indicador.upper()}"
    )

    fig.update_layout(
        xaxis_title=f"{indicador.upper()}",
        yaxis_title="ICV (Índice de Custo de Vida)"
    )

    st.plotly_chart(fig)





