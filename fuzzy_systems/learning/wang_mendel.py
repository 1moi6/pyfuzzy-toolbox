"""
Método de Wang-Mendel para Geração Automática de Regras Fuzzy
================================================================

Este módulo implementa o algoritmo de Wang-Mendel (1992) para gerar
automaticamente regras fuzzy a partir de dados.

Referência:
    Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by
    learning from examples." IEEE Transactions on Systems, Man, and
    Cybernetics, 22(6), 1414-1427.

O algoritmo possui 5 passos:
1. Particionar os domínios das variáveis (fuzzificação)
2. Gerar regras candidatas dos dados
3. Atribuir grau a cada regra
4. Resolver conflitos (manter regra com maior grau)
5. Criar sistema fuzzy final
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ..core.fuzzification import LinguisticVariable, FuzzySet
from ..inference.systems import MamdaniSystem


class WangMendelRegression:
    """
    Wang-Mendel para Regressão Fuzzy.

    Aceita qualquer sistema Mamdani (SISO, MISO, SIMO, MIMO) e aprende
    regras automaticamente a partir de dados de regressão.

    Exemplo:
        >>> import fuzzy_systems as fs
        >>> import numpy as np
        >>>
        >>> # Criar sistema SISO
        >>> system = fs.MamdaniSystem()
        >>> system.add_input('x', (0, 10))
        >>> system.add_output('y', (-1, 1))
        >>>
        >>> # Adicionar termos
        >>> for name, params in [('baixo', (0,0,5)), ('medio', (0,5,10)), ('alto', (5,10,10))]:
        >>>     system.add_term('x', name, 'triangular', params)
        >>>     system.add_term('y', name, 'triangular', (-1 if name=='baixo' else (0 if name=='medio' else 1),
        >>>                                                -1 if name=='baixo' else (0 if name=='medio' else 1),
        >>>                                                0 if name=='baixo' else (1 if name=='medio' else 1)))
        >>>
        >>> # Dados de treino
        >>> X_train = np.linspace(0, 10, 50).reshape(-1, 1)
        >>> y_train = np.sin(X_train).reshape(-1, 1)
        >>>
        >>> # Treinar
        >>> wm = fs.learning.WangMendelRegression(system, X_train, y_train)
        >>> system_trained = wm.fit()
        >>>
        >>> # Predizer
        >>> y_pred = wm.predict(X_test)
    """

    def __init__(self,
                 system: MamdaniSystem,
                 X_train: np.ndarray,
                 y_train: np.ndarray):
        """
        Inicializa o Wang-Mendel para regressão.

        Parâmetros:
            system: Sistema Mamdani (SISO, MISO, SIMO ou MIMO) já configurado
                   com variáveis e termos fuzzy
            X_train: Dados de entrada (n_amostras, n_entradas)
            y_train: Dados de saída (n_amostras, n_saidas)
        """
        self.system = system

        # Garantir formato 2D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self.X_train = X_train
        self.y_train = y_train

        # Extrair informações do sistema
        self.input_vars = list(system.input_variables.values())
        self.output_vars = list(system.output_variables.values())
        self.input_names = list(system.input_variables.keys())
        self.output_names = list(system.output_variables.keys())

        self.n_inputs = len(self.input_vars)
        self.n_outputs = len(self.output_vars)

        # Validações
        if X_train.shape[1] != self.n_inputs:
            raise ValueError(
                f"X_train tem {X_train.shape[1]} colunas, mas sistema tem {self.n_inputs} entradas"
            )
        if y_train.shape[1] != self.n_outputs:
            raise ValueError(
                f"y_train tem {y_train.shape[1]} colunas, mas sistema tem {self.n_outputs} saídas"
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train ({len(X_train)} amostras) e y_train ({len(y_train)} amostras) "
                "devem ter o mesmo número de amostras"
            )

        # Armazenamento de regras
        self.regras_dict_list = [{} for _ in range(self.n_outputs)]
        self.conflitos_list = [0] * self.n_outputs
        self._gerar_regras_done = False

    def fit(self, verbose: bool = True, n_examples: int = 5) -> MamdaniSystem:
        """
        Executa o algoritmo de Wang-Mendel.

        Passos:
        1. Gera regras a partir dos dados
        2. Resolve conflitos
        3. Adiciona regras ao sistema

        Parâmetros:
            verbose: Se True, imprime informações
            n_examples: Número de exemplos de regras a mostrar

        Retorna:
            O sistema Mamdani treinado
        """
        self.generate_rules(verbose=verbose, n_examples=n_examples)
        self.add_rules_to_system(verbose=verbose)
        return self.system

    def generate_rules(self, verbose: bool = True, n_examples: int = 5) -> None:
        """
        Gera regras para cada saída usando o algoritmo de Wang-Mendel.

        Para sistemas MIMO: Cada saída é tratada independentemente.
        Conflitos são resolvidos por saída.
        """
        if verbose:
            print("=" * 70)
            print("WANG-MENDEL REGRESSÃO - GERANDO REGRAS")
            print("=" * 70)
            print(f"\n📊 Configuração:")
            print(f"   • Entradas:  {self.n_inputs} ({self.input_names})")
            print(f"   • Saídas:    {self.n_outputs} ({self.output_names})")
            print(f"   • Amostras:  {len(self.X_train)}\n")

        # Resetar regras
        self.regras_dict_list = [{} for _ in range(self.n_outputs)]
        self.conflitos_list = [0] * self.n_outputs

        # Para cada amostra de treino
        for idx, (x_sample, y_sample) in enumerate(zip(self.X_train, self.y_train)):
            # ANTECEDENTE: Encontrar melhor termo para cada entrada
            best_terms = []
            max_mus = []

            for i, var in enumerate(self.input_vars):
                x_val = x_sample[i]
                max_mu = -1
                best_term = list(var.terms.keys())[0]  # Default: primeiro termo

                for term_name, fuzzy_set in var.terms.items():
                    mu = fuzzy_set.membership(x_val)
                    if mu > max_mu:
                        max_mu = mu
                        best_term = term_name

                best_terms.append(best_term)
                max_mus.append(max(max_mu, 0))

            # Calcular grau da regra
            grau_regra = np.prod(max_mus)
            chave = tuple(best_terms)

            # CONSEQUENTE: Para cada saída (independente)
            for out_idx, var_out in enumerate(self.output_vars):
                y_val = y_sample[out_idx]

                # Encontrar melhor termo de saída
                max_mu_out = -1
                output_term = list(var_out.terms.keys())[0]  # Default

                for term_name, fuzzy_set in var_out.terms.items():
                    mu = fuzzy_set.membership(y_val)
                    if mu > max_mu_out:
                        max_mu_out = mu
                        output_term = term_name

                # Verificar conflito
                if chave in self.regras_dict_list[out_idx]:
                    output_antigo, grau_antigo = self.regras_dict_list[out_idx][chave]
                    if grau_regra > grau_antigo:
                        self.regras_dict_list[out_idx][chave] = (output_term, grau_regra)
                        self.conflitos_list[out_idx] += 1
                    else:
                        self.conflitos_list[out_idx] += 1
                else:
                    self.regras_dict_list[out_idx][chave] = (output_term, grau_regra)

        # Verbose output
        if verbose:
            print("=" * 70)
            print("REGRAS GERADAS POR SAÍDA")
            print("=" * 70)
            for out_idx, (var_out, regras_dict) in enumerate(zip(self.output_vars, self.regras_dict_list)):
                print(f"\n📤 Saída: {var_out.name}")
                print(f"   ✅ Regras geradas: {len(regras_dict)}")
                print(f"   ⚠️  Conflitos resolvidos: {self.conflitos_list[out_idx]}")

                if n_examples > 0 and len(regras_dict) > 0:
                    print(f"\n   📋 Exemplos de regras (mostrando {min(n_examples, len(regras_dict))}):")
                    for i, (antecedente, (consequente, grau)) in enumerate(list(regras_dict.items())[:n_examples]):
                        condicoes = [f"{self.input_names[j]} = {termo}" for j, termo in enumerate(antecedente)]
                        antecedente_str = " AND ".join(condicoes)
                        print(f"      {i+1}. IF {antecedente_str}")
                        print(f"         THEN {var_out.name} = {consequente} (grau={grau:.3f})")

                    if len(regras_dict) > n_examples:
                        print(f"      ... (e mais {len(regras_dict) - n_examples} regras)")
            print()

        self._gerar_regras_done = True

    def add_rules_to_system(self, verbose: bool = True) -> MamdaniSystem:
        """
        Adiciona as regras geradas ao sistema fornecido.

        Para sistemas MIMO, cada regra deve ter consequentes para TODAS as saídas.
        Este método combina as regras de diferentes saídas com o mesmo antecedente.

        Parâmetros:
            verbose: Se True, imprime informações

        Retorna:
            O mesmo sistema, agora com as regras adicionadas
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute generate_rules() primeiro!")

        if verbose:
            print("=" * 70)
            print("ADICIONANDO REGRAS AO SISTEMA")
            print("=" * 70)
            print()

        # Limpar regras existentes
        self.system.rule_base.rules = []

        # Combinar regras de todas as saídas
        # Mapear: antecedente -> [consequente_saida_0, consequente_saida_1, ...]
        combined_rules = {}

        for out_idx, regras_dict in enumerate(self.regras_dict_list):
            for antecedente, (consequente, grau) in regras_dict.items():
                if antecedente not in combined_rules:
                    # Inicializar com None para todas as saídas
                    combined_rules[antecedente] = [None] * self.n_outputs

                # Adicionar consequente para esta saída
                combined_rules[antecedente][out_idx] = consequente

        # Adicionar regras ao sistema
        # ATENÇÃO: Só adicionamos regras que têm consequentes para TODAS as saídas
        total_regras = 0
        regras_ignoradas = 0

        for antecedente, consequentes in combined_rules.items():
            # Verificar se todas as saídas têm consequente
            if None in consequentes:
                regras_ignoradas += 1
                if verbose and regras_ignoradas <= 3:
                    print(f"   ⚠️  Ignorando regra incompleta: {antecedente} -> {consequentes}")
                continue

            # Criar regra completa
            antecedentes_lista = list(antecedente)
            self.system.add_rules([(antecedentes_lista, consequentes)])
            total_regras += 1

        if verbose:
            if regras_ignoradas > 3:
                print(f"   ⚠️  ... e mais {regras_ignoradas - 3} regras incompletas ignoradas")
            print()
            print(f"✅ {total_regras} regras completas adicionadas ao sistema!")
            if regras_ignoradas > 0:
                print(f"⚠️  {regras_ignoradas} regras incompletas foram ignoradas")
            print()
            print(f"   Entradas: {self.input_names}")
            print(f"   Saídas: {self.output_names}")

        return self.system

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições usando o sistema treinado.

        Parâmetros:
            X: Dados de entrada (n_amostras, n_entradas)

        Retorna:
            Predições (n_amostras, n_saidas)
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute fit() antes de predict()")

        # Garantir formato 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = []
        for x_sample in X:
            # Criar dicionário de entrada
            inputs = {self.input_names[i]: x_sample[i] for i in range(self.n_inputs)}

            # Avaliar sistema
            output = self.system.evaluate(inputs)

            # Extrair valores na ordem correta
            y_vals = [output[name] for name in self.output_names]
            y_pred.append(y_vals)

        return np.array(y_pred)

    def get_training_stats(self) -> Dict:
        """
        Retorna estatísticas do treinamento.

        Retorna:
            Dicionário com estatísticas:
            - n_samples: Número de amostras de treino
            - n_inputs: Número de entradas
            - n_outputs: Número de saídas
            - input_names: Nomes das entradas
            - output_names: Nomes das saídas
            - rules_per_output: Lista com número de regras por saída
            - conflicts_per_output: Lista com número de conflitos resolvidos por saída
            - total_rules: Total de regras combinadas adicionadas ao sistema
            - coverage: Cobertura (% de regras completas vs candidatas)

        Exemplo:
            >>> wm = WangMendelRegression(system, X_train, y_train)
            >>> wm.fit()
            >>> stats = wm.get_training_stats()
            >>> print(f"Regras geradas: {stats['total_rules']}")
            >>> print(f"Conflitos resolvidos: {sum(stats['conflicts_per_output'])}")
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute fit() antes de get_training_stats()")

        # Calcular total de regras candidatas
        total_candidatas = sum(len(regras) for regras in self.regras_dict_list)

        # Total de regras completas no sistema
        total_rules = len(self.system.rule_base.rules)

        # Cobertura
        coverage = (total_rules / total_candidatas * 100) if total_candidatas > 0 else 0

        return {
            'n_samples': len(self.X_train),
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'rules_per_output': [len(regras) for regras in self.regras_dict_list],
            'conflicts_per_output': self.conflitos_list,
            'total_rules': total_rules,
            'total_candidate_rules': total_candidatas,
            'coverage': coverage,
        }


class WangMendelClassification:
    """
    Wang-Mendel para Classificação Fuzzy.

    Aceita qualquer sistema Mamdani (SISO, MISO, SIMO, MIMO) e aprende
    regras automaticamente a partir de dados de classificação.

    Detecta automaticamente one-hot encoding quando aplicável.

    Exemplo:
        >>> import fuzzy_systems as fs
        >>> import numpy as np
        >>>
        >>> # Criar sistema para Iris (4 entradas, 3 saídas one-hot)
        >>> system = fs.MamdaniSystem()
        >>>
        >>> # Adicionar entradas
        >>> for name in ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']:
        >>>     system.add_input(name, (0, 10))
        >>>     for term in ['baixo', 'medio', 'alto']:
        >>>         system.add_term(name, term, 'triangular', (0, 5, 10))
        >>>
        >>> # Adicionar saídas (one-hot)
        >>> for name in ['setosa', 'versicolor', 'virginica']:
        >>>     system.add_output(name, (0, 1))
        >>>     system.add_term(name, 'nao', 'triangular', (-0.01, 0, 0.01))
        >>>     system.add_term(name, 'sim', 'triangular', (0.99, 1, 1.01))
        >>>
        >>> # Dados de treino (one-hot)
        >>> X_train = ...  # (n_amostras, 4)
        >>> y_train = ...  # (n_amostras, 3) com [1,0,0], [0,1,0], [0,0,1]
        >>>
        >>> # Treinar
        >>> wm = fs.learning.WangMendelClassification(system, X_train, y_train)
        >>> system_trained = wm.fit()
        >>>
        >>> # Predizer classes
        >>> y_pred_classes = wm.predict(X_test)  # Retorna índices das classes
    """

    def __init__(self,
                 system: MamdaniSystem,
                 X_train: np.ndarray,
                 y_train: np.ndarray):
        """
        Inicializa o Wang-Mendel para classificação.

        Parâmetros:
            system: Sistema Mamdani (SISO, MISO, SIMO ou MIMO) já configurado
                   com variáveis e termos fuzzy
            X_train: Dados de entrada (n_amostras, n_entradas)
            y_train: Dados de saída (n_amostras, n_saidas)
                    - Para classificação one-hot: valores 0/1 com soma=1 por linha
                    - Para classificação binária: valores 0/1
                    - Para multi-classe única saída: índices de classe (0, 1, 2, ...)
        """
        self.system = system

        # Garantir formato 2D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        self.X_train = X_train
        self.y_train = y_train

        # Extrair informações do sistema
        self.input_vars = list(system.input_variables.values())
        self.output_vars = list(system.output_variables.values())
        self.input_names = list(system.input_variables.keys())
        self.output_names = list(system.output_variables.keys())

        self.n_inputs = len(self.input_vars)
        self.n_outputs = len(self.output_vars)

        # Validações
        if X_train.shape[1] != self.n_inputs:
            raise ValueError(
                f"X_train tem {X_train.shape[1]} colunas, mas sistema tem {self.n_inputs} entradas"
            )
        if y_train.shape[1] != self.n_outputs:
            raise ValueError(
                f"y_train tem {y_train.shape[1]} colunas, mas sistema tem {self.n_outputs} saídas"
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train ({len(X_train)} amostras) e y_train ({len(y_train)} amostras) "
                "devem ter o mesmo número de amostras"
            )

        # Armazenamento de regras
        self.regras_dict_list = [{} for _ in range(self.n_outputs)]
        self.conflitos_list = [0] * self.n_outputs
        self._gerar_regras_done = False
        self._is_one_hot = False

    def fit(self, verbose: bool = True, n_examples: int = 5) -> MamdaniSystem:
        """
        Executa o algoritmo de Wang-Mendel para classificação.

        Passos:
        1. Detecta se é one-hot encoding
        2. Gera regras a partir dos dados
        3. Resolve conflitos
        4. Adiciona regras ao sistema

        Parâmetros:
            verbose: Se True, imprime informações
            n_examples: Número de exemplos de regras a mostrar

        Retorna:
            O sistema Mamdani treinado
        """
        self.generate_rules(verbose=verbose, n_examples=n_examples)
        self.add_rules_to_system(verbose=verbose)
        return self.system

    def generate_rules(self, verbose: bool = True, n_examples: int = 5) -> None:
        """
        Gera regras para cada saída usando o algoritmo de Wang-Mendel.

        Para sistemas MIMO, há dois modos de operação:

        1. Classificação independente: Cada saída é tratada separadamente.
           Conflitos são resolvidos por saída.

        2. Classificação One-Hot (detecção automática): Se os dados parecem one-hot
           (valores 0/1 e soma das linhas ≈ 1), trata como vetor único.
           Conflitos são resolvidos considerando o vetor completo.
        """
        if verbose:
            print("=" * 70)
            print("WANG-MENDEL CLASSIFICAÇÃO - GERANDO REGRAS")
            print("=" * 70)
            print(f"\n📊 Configuração:")
            print(f"   • Entradas:  {self.n_inputs} ({self.input_names})")
            print(f"   • Saídas:    {self.n_outputs} ({self.output_names})")
            print(f"   • Amostras:  {len(self.X_train)}\n")

        # Detectar se é one-hot encoding
        is_one_hot = False
        if self.n_outputs > 1:
            # Verificar se valores são binários (0 ou 1)
            is_binary = np.all(np.isin(self.y_train, [0, 1]))

            # Verificar se soma de cada linha é aproximadamente 1
            row_sums = np.sum(self.y_train, axis=1)
            sum_is_one = np.allclose(row_sums, 1.0, atol=0.1)

            if is_binary and sum_is_one:
                is_one_hot = True
                self._is_one_hot = True
                if verbose:
                    print("   🔍 Detectado: Classificação One-Hot Encoding")
                    print("      → Regras serão geradas considerando todas as saídas juntas\n")

        # Resetar regras
        self.regras_dict_list = [{} for _ in range(self.n_outputs)]
        self.conflitos_list = [0] * self.n_outputs

        if is_one_hot:
            # Dicionário compartilhado para regras one-hot
            regras_onehot = {}  # chave -> (lista_consequentes, grau)

        # Para cada amostra de treino
        for idx, (x_sample, y_sample) in enumerate(zip(self.X_train, self.y_train)):
            # ANTECEDENTE: Encontrar melhor termo para cada entrada
            best_terms = []
            max_mus = []

            for i, var in enumerate(self.input_vars):
                x_val = x_sample[i]
                max_mu = -1
                best_term = list(var.terms.keys())[0]  # Default: primeiro termo

                for term_name, fuzzy_set in var.terms.items():
                    mu = fuzzy_set.membership(x_val)
                    if mu > max_mu:
                        max_mu = mu
                        best_term = term_name

                best_terms.append(best_term)
                max_mus.append(max(max_mu, 0))

            # Calcular grau da regra
            grau_regra = np.prod(max_mus)
            chave = tuple(best_terms)

            if is_one_hot:
                # MODO ONE-HOT: Tratar todas as saídas juntas
                consequentes = []
                for out_idx, var_out in enumerate(self.output_vars):
                    y_val = y_sample[out_idx]
                    # Para one-hot, y_val é 0 ou 1
                    output_term = list(var_out.terms.keys())[int(y_val)]
                    consequentes.append(output_term)

                # Verificar conflito
                if chave in regras_onehot:
                    cons_antigo, grau_antigo = regras_onehot[chave]
                    if grau_regra > grau_antigo:
                        regras_onehot[chave] = (consequentes, grau_regra)
                        for out_idx in range(self.n_outputs):
                            self.conflitos_list[out_idx] += 1
                else:
                    regras_onehot[chave] = (consequentes, grau_regra)

            else:
                # MODO NORMAL: Cada saída independente
                for out_idx, var_out in enumerate(self.output_vars):
                    y_val = y_sample[out_idx]

                    # Para classificação, y_val é o índice da classe
                    output_term = list(var_out.terms.keys())[int(y_val)]

                    # Verificar conflito
                    if chave in self.regras_dict_list[out_idx]:
                        output_antigo, grau_antigo = self.regras_dict_list[out_idx][chave]
                        if grau_regra > grau_antigo:
                            self.regras_dict_list[out_idx][chave] = (output_term, grau_regra)
                            self.conflitos_list[out_idx] += 1
                        else:
                            self.conflitos_list[out_idx] += 1
                    else:
                        self.regras_dict_list[out_idx][chave] = (output_term, grau_regra)

        # Copiar regras one-hot para as listas individuais
        if is_one_hot:
            for chave, (consequentes, grau) in regras_onehot.items():
                for out_idx, cons in enumerate(consequentes):
                    self.regras_dict_list[out_idx][chave] = (cons, grau)

        # Verbose output
        if verbose:
            print("=" * 70)
            print("REGRAS GERADAS POR SAÍDA")
            print("=" * 70)
            for out_idx, (var_out, regras_dict) in enumerate(zip(self.output_vars, self.regras_dict_list)):
                print(f"\n📤 Saída: {var_out.name}")
                print(f"   ✅ Regras geradas: {len(regras_dict)}")
                print(f"   ⚠️  Conflitos resolvidos: {self.conflitos_list[out_idx]}")

                if n_examples > 0 and len(regras_dict) > 0:
                    print(f"\n   📋 Exemplos de regras (mostrando {min(n_examples, len(regras_dict))}):")
                    for i, (antecedente, (consequente, grau)) in enumerate(list(regras_dict.items())[:n_examples]):
                        condicoes = [f"{self.input_names[j]} = {termo}" for j, termo in enumerate(antecedente)]
                        antecedente_str = " AND ".join(condicoes)
                        print(f"      {i+1}. IF {antecedente_str}")
                        print(f"         THEN {var_out.name} = {consequente} (grau={grau:.3f})")

                    if len(regras_dict) > n_examples:
                        print(f"      ... (e mais {len(regras_dict) - n_examples} regras)")
            print()

        self._gerar_regras_done = True

    def add_rules_to_system(self, verbose: bool = True) -> MamdaniSystem:
        """
        Adiciona as regras geradas ao sistema fornecido.

        Para sistemas MIMO, cada regra deve ter consequentes para TODAS as saídas.
        Este método combina as regras de diferentes saídas com o mesmo antecedente.

        Parâmetros:
            verbose: Se True, imprime informações

        Retorna:
            O mesmo sistema, agora com as regras adicionadas
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute generate_rules() primeiro!")

        if verbose:
            print("=" * 70)
            print("ADICIONANDO REGRAS AO SISTEMA")
            print("=" * 70)
            print()

        # Limpar regras existentes
        self.system.rule_base.rules = []

        # Combinar regras de todas as saídas
        # Mapear: antecedente -> [consequente_saida_0, consequente_saida_1, ...]
        combined_rules = {}

        for out_idx, regras_dict in enumerate(self.regras_dict_list):
            for antecedente, (consequente, grau) in regras_dict.items():
                if antecedente not in combined_rules:
                    # Inicializar com None para todas as saídas
                    combined_rules[antecedente] = [None] * self.n_outputs

                # Adicionar consequente para esta saída
                combined_rules[antecedente][out_idx] = consequente

        # Adicionar regras ao sistema
        # ATENÇÃO: Só adicionamos regras que têm consequentes para TODAS as saídas
        total_regras = 0
        regras_ignoradas = 0

        for antecedente, consequentes in combined_rules.items():
            # Verificar se todas as saídas têm consequente
            if None in consequentes:
                regras_ignoradas += 1
                if verbose and regras_ignoradas <= 3:
                    print(f"   ⚠️  Ignorando regra incompleta: {antecedente} -> {consequentes}")
                continue

            # Criar regra completa
            antecedentes_lista = list(antecedente)
            self.system.add_rules([(antecedentes_lista, consequentes)])
            total_regras += 1

        if verbose:
            if regras_ignoradas > 3:
                print(f"   ⚠️  ... e mais {regras_ignoradas - 3} regras incompletas ignoradas")
            print()
            print(f"✅ {total_regras} regras completas adicionadas ao sistema!")
            if regras_ignoradas > 0:
                print(f"⚠️  {regras_ignoradas} regras incompletas foram ignoradas")
            print()
            print(f"   Entradas: {self.input_names}")
            print(f"   Saídas: {self.output_names}")

        return self.system

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições de classes usando o sistema treinado.

        Parâmetros:
            X: Dados de entrada (n_amostras, n_entradas)

        Retorna:
            Se one-hot: índices das classes (n_amostras,) obtidos por argmax
            Se não one-hot: valores de saída (n_amostras, n_saidas)
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute fit() antes de predict()")

        # Garantir formato 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred_scores = []
        for x_sample in X:
            # Criar dicionário de entrada
            inputs = {self.input_names[i]: x_sample[i] for i in range(self.n_inputs)}

            # Avaliar sistema
            output = self.system.evaluate(inputs)

            # Extrair valores na ordem correta
            y_vals = [output[name] for name in self.output_names]
            y_pred_scores.append(y_vals)

        y_pred_scores = np.array(y_pred_scores)

        # Se one-hot, retornar índices das classes (argmax)
        if self._is_one_hot:
            return np.argmax(y_pred_scores, axis=1)
        else:
            return y_pred_scores

    def predict_membership(self, X: np.ndarray) -> List[List[Dict[str, float]]]:
        """
        Faz predições retornando os graus de pertinência em cada termo linguístico.

        Este método é útil para entender a incerteza e confiança das predições,
        mostrando quanto cada valor de saída pertence a cada termo fuzzy.

        Parâmetros:
            X: Dados de entrada (n_amostras, n_entradas)

        Retorna:
            Lista de listas de dicionários:
            - Nível 1: uma lista por amostra
            - Nível 2: uma lista por saída
            - Nível 3: dicionário {termo: grau_pertinência}

            Exemplo para 2 amostras, 3 saídas (one-hot):
            [
                [  # Amostra 1
                    {'nao': 0.8, 'sim': 0.2},  # class0
                    {'nao': 0.1, 'sim': 0.9},  # class1
                    {'nao': 0.7, 'sim': 0.3}   # class2
                ],
                [  # Amostra 2
                    {'nao': 0.2, 'sim': 0.8},  # class0
                    {'nao': 0.9, 'sim': 0.1},  # class1
                    {'nao': 0.6, 'sim': 0.4}   # class2
                ]
            ]

        Exemplo:
            >>> wm = WangMendelClassification(system, X_train, y_train)
            >>> wm.fit()
            >>> memberships = wm.predict_membership(X_test)
            >>>
            >>> # Para primeira amostra, primeira saída:
            >>> print(memberships[0][0])
            {'nao': 0.8, 'sim': 0.2}
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute fit() antes de predict_membership()")

        # Garantir formato 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        all_memberships = []

        for x_sample in X:
            # Criar dicionário de entrada
            inputs = {self.input_names[i]: x_sample[i] for i in range(self.n_inputs)}

            # Avaliar sistema
            output = self.system.evaluate(inputs)

            # Para cada saída, calcular pertinência em cada termo
            sample_memberships = []
            for out_name in self.output_names:
                y_val = output[out_name]
                var_out = self.system.output_variables[out_name]

                # Calcular pertinência para cada termo desta saída
                term_memberships = {}
                for term_name, fuzzy_set in var_out.terms.items():
                    membership = fuzzy_set.membership(y_val)
                    term_memberships[term_name] = float(membership)

                sample_memberships.append(term_memberships)

            all_memberships.append(sample_memberships)

        return all_memberships

    def get_training_stats(self) -> Dict:
        """
        Retorna estatísticas do treinamento.

        Retorna:
            Dicionário com estatísticas:
            - n_samples: Número de amostras de treino
            - n_inputs: Número de entradas
            - n_outputs: Número de saídas
            - input_names: Nomes das entradas
            - output_names: Nomes das saídas
            - is_one_hot: Se foi detectado one-hot encoding
            - rules_per_output: Lista com número de regras por saída
            - conflicts_per_output: Lista com número de conflitos resolvidos por saída
            - total_rules: Total de regras combinadas adicionadas ao sistema
            - coverage: Cobertura (% de regras completas vs candidatas)
            - class_distribution: Distribuição das classes no treino (para one-hot)

        Exemplo:
            >>> wm = WangMendelClassification(system, X_train, y_train)
            >>> wm.fit()
            >>> stats = wm.get_training_stats()
            >>> print(f"One-hot: {stats['is_one_hot']}")
            >>> print(f"Regras: {stats['total_rules']}")
            >>> print(f"Distribuição: {stats['class_distribution']}")
        """
        if not self._gerar_regras_done:
            raise RuntimeError("Execute fit() antes de get_training_stats()")

        # Calcular total de regras candidatas
        total_candidatas = sum(len(regras) for regras in self.regras_dict_list)

        # Total de regras completas no sistema
        total_rules = len(self.system.rule_base.rules)

        # Cobertura
        coverage = (total_rules / total_candidatas * 100) if total_candidatas > 0 else 0

        # Distribuição de classes (para one-hot)
        class_distribution = None
        if self._is_one_hot:
            # Contar quantas amostras por classe
            class_counts = np.sum(self.y_train, axis=0)
            class_distribution = {
                self.output_names[i]: int(class_counts[i])
                for i in range(self.n_outputs)
            }

        return {
            'n_samples': len(self.X_train),
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'is_one_hot': self._is_one_hot,
            'rules_per_output': [len(regras) for regras in self.regras_dict_list],
            'conflicts_per_output': self.conflitos_list,
            'total_rules': total_rules,
            'total_candidate_rules': total_candidatas,
            'coverage': coverage,
            'class_distribution': class_distribution,
        }
