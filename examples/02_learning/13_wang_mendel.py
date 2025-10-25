"""
Exemplo 13: Método de Wang-Mendel para Geração Automática de Regras

Este exemplo demonstra o uso do algoritmo de Wang-Mendel para gerar
automaticamente regras fuzzy a partir de dados de entrada/saída.

O Wang-Mendel é um dos algoritmos mais populares para aprendizado de regras fuzzy,
especialmente útil quando não se conhece previamente a estrutura das regras.

Referência:
    Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by learning from examples."
    IEEE Transactions on Systems, Man, and Cybernetics, 22(6), 1414-1427.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fuzzy_systems as fs


def exemplo_1_regressao_simples():
    """
    Exemplo 1: Regressão simples - aproximar função seno
    """
    print("=" * 70)
    print("EXEMPLO 1: Regressão Simples - Aproximação de Função")
    print("=" * 70)
    print()

    # Gerar dados de treino
    print("1. GERANDO DADOS DE TREINO")
    print("-" * 70)
    np.random.seed(42)
    X_train = np.linspace(0, 2*np.pi, 20)
    y_train = np.sin(X_train)

    print(f"   Amostras de treino: {len(X_train)}")
    print(f"   Função: y = sin(x)")
    print(f"   Domínio: [0, 2π]")
    print()

    # Criar e treinar modelo Wang-Mendel
    print("2. TREINANDO MODELO WANG-MENDEL")
    print("-" * 70)
    wm = fs.learning.WangMendel(
        X_train=X_train,
        y_train=y_train,
        n_partitions=7,
        margin=0.10,
        input_names=['x'],
        output_name='sin(x)'
    )

    # Executar algoritmo (mostra os 5 passos)
    system = wm.fit(verbose=True)

    # Fazer predições
    print("\n3. FAZENDO PREDIÇÕES")
    print("-" * 70)
    X_test = np.linspace(0, 2*np.pi, 100)
    y_pred = wm.predict(X_test)
    y_true = np.sin(X_test)

    # Calcular erro
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    print(f"\n   Métricas no conjunto de teste:")
    print(f"   • MSE (Erro Quadrático Médio):  {mse:.6f}")
    print(f"   • MAE (Erro Absoluto Médio):    {mae:.6f}")
    print(f"   • Erro Máximo:                  {max_error:.6f}")

    # Mostrar algumas predições
    print(f"\n   Exemplos de predições:")
    indices = [0, 25, 50, 75, 99]
    for idx in indices:
        print(f"   x = {X_test[idx]:.3f} → predito = {y_pred[idx]:.3f}, real = {y_true[idx]:.3f}")

    # Informações do modelo
    print("\n4. INFORMAÇÕES DO MODELO")
    print("-" * 70)
    wm.info()

    # Mostrar regras descartadas (se houver)
    print("\n5. REGRAS DESCARTADAS")
    print("-" * 70)
    wm.show_discarded_rules(n_examples=3)

    return system


def exemplo_2_regressao_multivariavel():
    """
    Exemplo 2: Regressão multivariável - função de 2 variáveis
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Regressão Multivariável (2 entradas)")
    print("=" * 70)
    print()

    # Gerar dados: z = x^2 + y^2
    print("1. GERANDO DADOS")
    print("-" * 70)
    np.random.seed(123)
    n_samples = 50

    X_train = np.random.rand(n_samples, 2) * 4 - 2  # [-2, 2]
    y_train = X_train[:, 0]**2 + X_train[:, 1]**2

    print(f"   Amostras: {n_samples}")
    print(f"   Função: z = x² + y²")
    print(f"   Domínio: x, y ∈ [-2, 2]")
    print()

    # Treinar Wang-Mendel
    print("2. TREINANDO WANG-MENDEL")
    print("-" * 70)
    wm = fs.learning.WangMendel(
        X_train=X_train,
        y_train=y_train,
        n_partitions=5,
        input_names=['x', 'y'],
        output_name='z'
    )

    system = wm.fit(verbose=True)

    # Testar
    print("\n3. TESTANDO MODELO")
    print("-" * 70)
    X_test = np.array([
        [0, 0],      # z = 0
        [1, 1],      # z = 2
        [-1, 1],     # z = 2
        [2, 0],      # z = 4
        [1.5, 1.5]   # z = 4.5
    ])

    y_pred = wm.predict(X_test)
    y_true = X_test[:, 0]**2 + X_test[:, 1]**2

    print("\n   Comparação:")
    print("   " + "-" * 50)
    for i in range(len(X_test)):
        print(f"   x={X_test[i, 0]:5.1f}, y={X_test[i, 1]:5.1f} → "
              f"predito={y_pred[i]:5.2f}, real={y_true[i]:5.2f}, "
              f"erro={abs(y_pred[i]-y_true[i]):5.2f}")

    mse = np.mean((y_true - y_pred)**2)
    print(f"\n   MSE: {mse:.4f}")


def exemplo_3_classificacao():
    """
    Exemplo 3: Classificação - Iris simplificado
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Classificação - Dataset Íris (simplificado)")
    print("=" * 70)
    print()

    # Criar dados sintéticos inspirados no Iris
    print("1. GERANDO DADOS DE CLASSIFICAÇÃO")
    print("-" * 70)
    np.random.seed(42)

    # Classe 0: setosa (pequenas)
    X_class0 = np.random.randn(20, 2) * 0.3 + np.array([1, 0.5])

    # Classe 1: versicolor (médias)
    X_class1 = np.random.randn(20, 2) * 0.4 + np.array([3, 2])

    # Classe 2: virginica (grandes)
    X_class2 = np.random.randn(20, 2) * 0.5 + np.array([5, 4])

    X_train = np.vstack([X_class0, X_class1, X_class2])
    y_train = np.array([0]*20 + [1]*20 + [2]*20)

    # Embaralhar
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"   Amostras por classe:")
    print(f"   • Classe 0 (setosa):     20 amostras")
    print(f"   • Classe 1 (versicolor): 20 amostras")
    print(f"   • Classe 2 (virginica):  20 amostras")
    print(f"   Total: {len(X_train)} amostras")
    print()

    # Treinar classificador Wang-Mendel
    print("2. TREINANDO CLASSIFICADOR")
    print("-" * 70)
    wm = fs.learning.WangMendel(
        X_train=X_train,
        y_train=y_train,
        n_partitions=5,
        input_names=['comprimento_petala', 'largura_petala'],
        output_name='especie',
        classification=True,
        output_labels=['setosa', 'versicolor', 'virginica']
    )

    system = wm.fit(verbose=True)

    # Testar
    print("\n3. TESTANDO CLASSIFICADOR")
    print("-" * 70)

    # Criar casos de teste
    X_test = np.array([
        [1.0, 0.5],   # Deve ser setosa
        [3.0, 2.0],   # Deve ser versicolor
        [5.0, 4.0],   # Deve ser virginica
        [2.0, 1.5],   # Entre setosa e versicolor
        [4.0, 3.0]    # Entre versicolor e virginica
    ])

    y_pred = wm.predict(X_test)
    labels = ['setosa', 'versicolor', 'virginica']

    print("\n   Predições:")
    print("   " + "-" * 60)
    for i in range(len(X_test)):
        print(f"   Amostra {i+1}: comprimento={X_test[i, 0]:.2f}, largura={X_test[i, 1]:.2f}")
        print(f"              → Classe predita: {labels[y_pred[i]]}")
        print()

    # Acurácia no treino
    y_train_pred = wm.predict(X_train)
    accuracy = np.mean(y_train_pred == y_train) * 100
    print(f"   Acurácia no treino: {accuracy:.1f}%")


def exemplo_4_particoes_customizadas():
    """
    Exemplo 4: Usando número diferente de partições por variável
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Partições Customizadas por Variável")
    print("=" * 70)
    print()

    print("Cenário: Variáveis com diferentes granularidades")
    print("-" * 70)
    print("• temperatura (0-100): mais variação → 7 partições")
    print("• pressão (0-10): menos variação → 3 partições")
    print()

    # Gerar dados
    np.random.seed(999)
    n_samples = 40

    # temperatura tem mais variação (0-100)
    temp = np.random.rand(n_samples) * 100

    # pressão tem menos variação (0-10)
    pressao = np.random.rand(n_samples) * 10

    X_train = np.column_stack([temp, pressao])

    # Saída: fluxo = 0.5*temp + 2*pressao + ruído
    y_train = 0.5 * temp + 2 * pressao + np.random.randn(n_samples) * 2

    print("1. TREINANDO COM PARTIÇÕES DIFERENTES")
    print("-" * 70)

    wm = fs.learning.WangMendel(
        X_train=X_train,
        y_train=y_train,
        n_partitions_input=[7, 3],  # 7 para temperatura, 3 para pressão
        n_partitions_output=5,
        input_names=['temperatura', 'pressao'],
        output_name='fluxo'
    )

    system = wm.fit(verbose=True)

    print("\n2. TESTANDO")
    print("-" * 70)

    X_test = np.array([
        [20, 2],
        [50, 5],
        [80, 8]
    ])

    y_pred = wm.predict(X_test)
    y_true = 0.5 * X_test[:, 0] + 2 * X_test[:, 1]

    print("\n   Comparação:")
    for i in range(len(X_test)):
        print(f"   temp={X_test[i, 0]:.0f}, press={X_test[i, 1]:.0f} → "
              f"pred={y_pred[i]:.2f}, real={y_true[i]:.2f}")


def exemplo_5_exportar_importar_regras():
    """
    Exemplo 5: Exportar regras geradas pelo Wang-Mendel e reutilizá-las
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Exportar e Reutilizar Regras do Wang-Mendel")
    print("=" * 70)
    print()

    # Gerar dados
    np.random.seed(100)
    X_train = np.linspace(0, 10, 25).reshape(-1, 1)
    y_train = np.sin(X_train.ravel()) * 5 + 5  # Deslocado para [0, 10]

    print("1. TREINAR WANG-MENDEL")
    print("-" * 70)
    wm = fs.learning.WangMendel(
        X_train=X_train,
        y_train=y_train,
        n_partitions=5,
        input_names=['x'],
        output_name='y'
    )

    system = wm.fit(verbose=False)
    print(f"✅ Sistema treinado com {len(system.rule_base.rules)} regras")
    print()

    # Exportar regras
    print("2. EXPORTAR REGRAS")
    print("-" * 70)
    export_dir = os.path.join(os.path.dirname(__file__), 'exported_rules')
    os.makedirs(export_dir, exist_ok=True)

    rules_file = os.path.join(export_dir, 'wang_mendel_rules.json')
    system.export_rules(rules_file, format='json')
    print(f"✅ Regras exportadas para: {rules_file}")
    print()

    # Ver as regras
    print("3. VISUALIZAR REGRAS GERADAS")
    print("-" * 70)
    system.print_rules(style='compact', show_stats=False)
    print()

    # Mostrar estatísticas
    stats = system.rules_statistics()
    print("4. ESTATÍSTICAS DAS REGRAS")
    print("-" * 70)
    print(f"   Total de regras:     {stats['total']}")
    print(f"   Operadores:          {stats['by_operator']}")
    print(f"   Média antecedentes:  {stats['avg_antecedents']:.1f}")
    print(f"   Média consequentes:  {stats['avg_consequents']:.1f}")


def main():
    print("\n")
    print("*" * 70)
    print(" " * 18 + "MÉTODO DE WANG-MENDEL")
    print(" " * 12 + "Geração Automática de Regras Fuzzy")
    print("*" * 70)

    # Exemplo 1: Regressão simples
    exemplo_1_regressao_simples()

    input("\n\nPressione Enter para continuar para o Exemplo 2...")

    # Exemplo 2: Regressão multivariável
    exemplo_2_regressao_multivariavel()

    input("\n\nPressione Enter para continuar para o Exemplo 3...")

    # Exemplo 3: Classificação
    exemplo_3_classificacao()

    input("\n\nPressione Enter para continuar para o Exemplo 4...")

    # Exemplo 4: Partições customizadas
    exemplo_4_particoes_customizadas()

    input("\n\nPressione Enter para continuar para o Exemplo 5...")

    # Exemplo 5: Exportar/importar regras
    exemplo_5_exportar_importar_regras()

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO: MÉTODO DE WANG-MENDEL")
    print("=" * 70)

    print("""
✅ ALGORITMO DE WANG-MENDEL IMPLEMENTADO:

📖 O QUE É:
   O método de Wang-Mendel é um algoritmo para construção automática
   de bases de regras fuzzy a partir de dados de entrada/saída.

🔄 OS 5 PASSOS DO ALGORITMO:
   1. Particionar domínios das variáveis (fuzzificação)
   2. Gerar regras candidatas dos dados
   3. Atribuir grau a cada regra (produto das pertinências)
   4. Resolver conflitos (manter regra com maior grau)
   5. Criar sistema fuzzy final

💡 VANTAGENS:
   ✓ Não requer conhecimento prévio das regras
   ✓ Geração automática a partir de dados
   ✓ Funciona para regressão e classificação
   ✓ Suporta múltiplas entradas
   ✓ Interpretável (regras linguísticas)
   ✓ Rápido e eficiente

🎯 APLICAÇÕES:
   • Aproximação de funções
   • Modelagem de sistemas complexos
   • Classificação de padrões
   • Controle automático
   • Sistemas especialistas
   • Interpolação de dados

⚙️  PARÂMETROS PRINCIPAIS:
   • n_partitions: Número de partições fuzzy (3, 5, 7, ...)
   • margin: Margem nos limites do universo (padrão: 15%)
   • classification: True para classificação, False para regressão
   • output_labels: Classes para classificação

📊 FUNCIONALIDADES EXTRAS:
   • Visualização de regras descartadas
   • Informações detalhadas do modelo
   • Exportação de regras para reutilização
   • Partições customizadas por variável

🔗 INTEGRAÇÃO COM O PACOTE:
   • Gera sistemas MamdaniSystem completos
   • Compatible com todas as funções de pertinência
   • Pode exportar/importar regras (JSON, CSV)
   • Visualização com plot_variables(), plot_output()
    """)

    print("=" * 70)
    print("FIM DO EXEMPLO")
    print("=" * 70)


if __name__ == "__main__":
    main()
