import logging
import os
from dotenv import load_dotenv
from supabase import create_client


class Sup_Cliente:
    _instancia = None  # Atributo de classe para armazenar a instância única

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(Sup_Cliente, cls).__new__(cls)
            load_dotenv()
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            cls._instancia.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return cls._instancia

    def rpc_no_param(self, function_name):
        response = self.supabase.rpc(function_name).execute()
        return response

    def rpc(self, function_name, params):
        # Acessa o cliente supabase e chama o método rpc
        response = self.supabase.rpc(function_name, params).execute()
        return response

    def salvar(self, tabela, dados):
        try:
            response = self.supabase.table(tabela).insert(dados).execute()

            # Verifica se a resposta contém dados válidos
            if hasattr(response, "data") and response.data:
                return response.data  # Retorna os dados inseridos
            elif hasattr(response, "error") and response.error:
                print(f"⚠️ Erro ao salvar os dados em '{tabela}': {response.error}")
                return None  # Retorna None se houver erro
            else:
                print(f"⚠️ Resposta inesperada ao salvar os dados em '{tabela}': {response}")
                return None

        except Exception as e:
            print(f"⚠️ Ocorreu um erro ao tentar salvar os dados na tabela '{tabela}': {e}")
            return None

    def buscar(self, tabela, filtros=None, ordenar_por=None, ordem="asc", limite=None):
        try:
            query = self.supabase.table(tabela).select("*")  # Inicia a query

            # Aplica os filtros, se existirem
            if filtros:
                for campo, valor in filtros.items():
                    query = query.eq(campo, valor)  # Usa eq() para aplicar os filtros

            # Adiciona ordenação, se especificada
            if ordenar_por:
                query = query.order(ordenar_por, desc=(ordem == "desc"))

            # Limita os resultados, se especificado
            if limite:
                query = query.limit(limite)

            # Executa a consulta
            response = query.execute()

            # Verifica se houve erro na resposta
            if hasattr(response, "error") and response.error:
                print(f"⚠️ Erro ao buscar dados: {response.error}")
                return []  # Retorna uma lista vazia em caso de erro

            # Retorna os dados encontrados ou uma lista vazia se não houver dados
            return response.data if hasattr(response, "data") and response.data else []

        except Exception as e:
            print(f"⚠️ Ocorreu um erro ao tentar buscar os dados: {e}")
            return []



    def buscar_por_colunas(self, tabela, colunas=None, filtros=None):
            """
            Busca registros no banco de dados.

            :param tabela: Nome da tabela no banco de dados.
            :param colunas: Lista de colunas a serem retornadas (se None, retorna todas).
            :param filtros: Dicionário de filtros no formato {"coluna": valor}.
            :return: Lista de dicionários representando os registros encontrados.
            """
            query = self.supabase.table(tabela).select(",".join(colunas) if colunas else "*")

            # Aplicar filtros se existirem

            if filtros and isinstance(filtros, dict):
                for coluna, valor in filtros.items():
                    query = query.eq(coluna, valor)

            resposta = query.execute()

            if resposta.data:
                return resposta.data
            else:
                return []  # Retorna uma lista vazia se não encontrar resultados


    def atualizar(self, tabela, colunas=None, filtros=None):

        query = self.supabase.table(tabela).update(colunas)

        # Aplicando filtros dinamicamente
        for coluna, valor in filtros.items():
            query = query.eq(coluna, valor)

        return query.execute()

    def buscar_in(self, tabela, filtros=None, ordenar_por=None, ordem="asc", limite=None):
        try:
            query = self.supabase.table(tabela).select("*")  # Inicia a query

            if filtros:
                for campo, valor in filtros.items():
                    # Se o valor for um dicionário com operador (como {"in": [...]})
                    if isinstance(valor, dict):
                        for operador, real_valor in valor.items():
                            if operador == "in":
                                query = query.in_(campo, real_valor)
                            elif operador == "gte":
                                query = query.gte(campo, real_valor)
                            elif operador == "lte":
                                query = query.lte(campo, real_valor)
                            elif operador == "neq":
                                query = query.neq(campo, real_valor)
                            elif operador == "like":
                                query = query.like(campo, real_valor)
                            else:
                                print(f"⚠️ Operador não suportado: {operador}")
                    else:
                        query = query.eq(campo, valor)

            if ordenar_por:
                query = query.order(ordenar_por, desc=(ordem == "desc"))

            if limite:
                query = query.limit(limite)

            response = query.execute()

            if hasattr(response, "error") and response.error:
                print(f"⚠️ Erro ao buscar dados: {response.error}")
                return []

            return response.data if hasattr(response, "data") and response.data else []

        except Exception as e:
            print(f"⚠️ Ocorreu um erro ao tentar buscar os dados: {e}")
            return []

    def buscar_comparando(self, tabela: str, filtros: list[tuple[str, str, any]]):
        """
        filtros: lista de tuplas no formato (campo, operador, valor)
        """
        query = self.supabase.table(tabela).select("*")

        for campo, op, valor in filtros:
            if op == "eq":
                query = query.eq(campo, valor)
            elif op == "neq":
                query = query.neq(campo, valor)
            elif op == "gte":
                query = query.gte(campo, valor)
            elif op == "lte":
                query = query.lte(campo, valor)
            elif op == "gt":
                query = query.gt(campo, valor)
            elif op == "lt":
                query = query.lt(campo, valor)
            elif op == "like":
                query = query.like(campo, valor)
            else:
                raise ValueError(f"Operador '{op}' não suportado.")

        resultado = query.execute()
        if resultado.data:
            return resultado.data
        return []
