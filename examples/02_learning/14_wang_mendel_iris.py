"""
Exemplo 14: Wang-Mendel para Classificação Iris - Duas Abordagens

Este exemplo demonstra duas formas de usar Wang-Mendel para classificação multi-classe:

1. ABORDAGEM 1 (Recomendada): 1 saída com 3 classes
   - Mais simples e direto
   - WangMendel cria automaticamente

2. ABORDAGEM 2 (One-vs-All): 3 saídas binárias (uma por classe)
   - Mais flexível para problemas complexos
   - Usa WangMendelFIS com MIMO
   - Decisão final por argmax()
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fuzzy_systems as fs



def gerar_dados_iris_sintetico():
    """
    Gera dados sintéticos inspirados no dataset Iris.

    Retorna:
        X: Features (n_amostras, 4) - comprimento_sepala, largura_sepala, comprimento_petala, largura_petala
        y: Classes (n_amostras,) - 0=setosa, 1=versicolor, 2=virginica
    """
    np.random.seed(42)

    # Classe 0: Iris Setosa (pequenas)
    n_samples = 30
    setosa = np.random.randn(n_samples, 4) * 0.3 + np.array([5.0, 3.5, 1.5, 0.3])

    # Classe 1: Iris Versicolor (médias)
    versicolor = np.random.randn(n_samples, 4) * 0.4 + np.array([6.0, 2.8, 4.5, 1.4])

    # Classe 2: Iris Virginica (grandes)
    virginica = np.random.randn(n_samples, 4) * 0.5 + np.array([6.5, 3.0, 5.5, 2.0])

    X = np.vstack([setosa, versicolor, virginica])
    y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

    # Embaralhar
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def abordagem_1_saida_unica():
    """
    ABORDAGEM 1: Uma saída com 3 classes (0, 1, 2)

    Vantagens:
    - Simples de implementar
    - WangMendel cuida de tudo automaticamente
    - Menos regras geradas

    Desvantagens:
    - Menos controle sobre as partições
    """
    print("=" * 70)
    print("ABORDAGEM 1: UMA SAÍDA COM 3 CLASSES")
    print("=" * 70)
    print()

    # 1. Gerar dados
    print("1. GERANDO DADOS IRIS")
    print("-" * 70)
    X, y = gerar_dados_iris_sintetico()

    print(f"   Amostras: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Distribuição: {[np.sum(y==i) for i in range(3)]}")
    print()

    # 2. Treinar classificador Wang-Mendel
    print("2. TREINANDO WANG-MENDEL")
    print("-" * 70)

    wm = fs.learning.WangMendelClassification(
        X_train=X,
        y_train=y,
        n_partitions=3,  # 3 partições por feature
        input_names=['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala'],
        output_name='especie',
        classification=True,
        output_labels=['setosa', 'versicolor', 'virginica']
    )

    system = wm.fit(verbose=False)

    print(f"   ✅ Sistema treinado com {len(system.rule_base.rules)} regras")
    print()

    # 3. Fazer predições
    print("3. TESTANDO CLASSIFICADOR")
    print("-" * 70)

    # Casos de teste
    X_test = np.array([
        [5.0, 3.5, 1.5, 0.3],   # Deve ser setosa
        [6.0, 2.8, 4.5, 1.4],   # Deve ser versicolor
        [6.5, 3.0, 5.5, 2.0],   # Deve ser virginica
    ])

    y_pred = wm.predict(X_test)
    labels = ['setosa', 'versicolor', 'virginica']

    print("\n   Predições:")
    for i in range(len(X_test)):
        print(f"   Amostra {i+1}: {X_test[i]}")
        print(f"              → Classe: {labels[y_pred[i]]}")
        print()

    # 4. Acurácia no treino
    y_train_pred = wm.predict(X)
    accuracy = np.mean(y_train_pred == y) * 100

    print(f"   Acurácia no treino: {accuracy:.1f}%")
    print()

    # 5. Mostrar algumas regras
    print("4. EXEMPLOS DE REGRAS GERADAS")
    print("-" * 70)
    system.print_rules(style='if-then', show_stats=False)

    return wm, system


def abordagem_2_one_vs_all():
    """
    ABORDAGEM 2: Três saídas binárias (one-vs-all)

    Vantagens:
    - Total controle sobre cada classificador binário
    - Pode ajustar partições diferentes para cada classe
    - Útil quando classes são muito desbalanceadas

    Desvantagens:
    - Mais complexo de implementar
    - Mais regras no total
    - Precisa de decisão final (argmax)
    """
    print("\n" + "=" * 70)
    print("ABORDAGEM 2: TRÊS SAÍDAS BINÁRIAS (ONE-VS-ALL)")
    print("=" * 70)
    print()

    # 1. Gerar dados
    print("1. GERANDO DADOS")
    print("-" * 70)
    X, y = gerar_dados_iris_sintetico()

    # Converter para formato one-vs-all: [eh_setosa, eh_versicolor, eh_virginica]
    y_train_onehot = np.zeros((len(y), 3))
    for i, classe in enumerate(y):
        y_train_onehot[i, classe] = 1

    print(f"   Amostras: {len(X)}")
    print(f"   Formato de y_train: {y_train_onehot.shape}")
    print(f"   Exemplo: classe=1 (versicolor) → {y_train_onehot[y==1][0]}")
    print()

    # 2. Criar sistema FIS com 3 saídas
    print("2. CRIANDO SISTEMA FIS COM 3 SAÍDAS")
    print("-" * 70)

    system = fs.MamdaniSystem(name="Iris One-vs-All")

    # Adicionar entradas
    input_names = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala']
    universes = [
        (np.min(X[:,0]), np.max(X[:,0])),  # comprimento sépala
        (np.min(X[:,1]), np.max(X[:,1])),  # largura sépala
        (np.min(X[:,2]), np.max(X[:,2])),  # comprimento pétala
        (np.min(X[:,3]), np.max(X[:,3])),  # largura pétala
    ]

    for name, universe in zip(input_names, universes):
        system.add_input(name, universe)
        # 3 partições por feature
        system.add_term(name, 'pequeno', 'triangular',
                       (universe[0], universe[0], (universe[0]+universe[1])/2))
        system.add_term(name, 'medio', 'triangular',
                       (universe[0], (universe[0]+universe[1])/2, universe[1]))
        system.add_term(name, 'grande', 'triangular',
                       ((universe[0]+universe[1])/2, universe[1], universe[1]))

    # Adicionar 3 saídas binárias
    output_names = ['eh_setosa', 'eh_versicolor', 'eh_virginica']
    for name in output_names:
        system.add_output(name, (0, 1))
        system.add_term(name, 'nao', 'triangular', (0,0,1.0))
        system.add_term(name, 'sim', 'triangular', (0, 1, 1.0))

    print(f"   ✅ Sistema criado:")
    print(f"      Entradas: {list(system.input_variables.keys())}")
    print(f"      Saídas: {list(system.output_variables.keys())}")
    print()

    # 3. Treinar com Wang-Mendel
    print("3. GERANDO REGRAS COM WANG-MENDEL")
    print("-" * 70)

    wm = fs.learning.WangMendelClassification(
        system=system,
        X_train=X,
        y_train=y_train_onehot,
    )

    system_trained = wm.fit(verbose=False)

    print(f"   ✅ Sistema treinado com {len(system_trained.rule_base.rules)} regras")

    # Verificar se há regras suficientes
    if len(system_trained.rule_base.rules) == 0:
        print()
        print("   ⚠️  ATENÇÃO: Nenhuma regra completa foi gerada!")
        print("   Isso acontece porque Wang-Mendel precisa que TODAS as saídas")
        print("   tenham consequente para o mesmo antecedente.")
        print()
        print("   💡 SOLUÇÃO: Use a Abordagem 1 para classificação multi-classe")
        print("   ou treine 3 classificadores binários separados (um por classe)")
        return None, None

    print()

    # 4. Fazer predições e decidir classe
    print("4. TESTANDO COM DECISÃO ARGMAX")
    print("-" * 70)

    X_test = np.array([
        [5.0, 3.5, 1.5, 0.3],   # Setosa
        [6.0, 2.8, 4.5, 1.4],   # Versicolor
        [6.5, 3.0, 5.5, 2.0],   # Virginica
    ])

    labels = ['setosa', 'versicolor', 'virginica']

    print("\n   Predições:")
    for i in range(len(X_test)):
        # Criar entrada
        inputs = {
            'comp_sepala': X_test[i, 0],
            'larg_sepala': X_test[i, 1],
            'comp_petala': X_test[i, 2],
            'larg_petala': X_test[i, 3]
        }

        # Obter saídas
        output = system_trained.evaluate(inputs)

        # Scores para cada classe
        scores = [
            output['eh_setosa'],
            output['eh_versicolor'],
            output['eh_virginica']
        ]

        # Decisão: argmax
        classe_predita = np.argmax(scores)

        print(f"   Amostra {i+1}: {X_test[i]}")
        print(f"      Scores: setosa={scores[0]:.3f}, versicolor={scores[1]:.3f}, virginica={scores[2]:.3f}")
        print(f"      → Classe: {labels[classe_predita]} (índice {classe_predita})")
        print()

    # 5. Acurácia no treino
    print("5. ACURÁCIA NO TREINO")
    print("-" * 70)

    y_pred = []
    for x_sample in X:
        inputs = {
            'comp_sepala': x_sample[0],
            'larg_sepala': x_sample[1],
            'comp_petala': x_sample[2],
            'larg_petala': x_sample[3]
        }
        output = system_trained.evaluate(inputs)
        scores = [output['eh_setosa'], output['eh_versicolor'], output['eh_virginica']]
        y_pred.append(np.argmax(scores))

    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y) * 100

    print(f"   Acurácia: {accuracy:.1f}%")
    print()

    return wm, system_trained


def comparacao_abordagens():
    """
    Compara as duas abordagens
    """
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DAS ABORDAGENS")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                   ABORDAGEM 1 vs ABORDAGEM 2                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ABORDAGEM 1: Uma Saída com 3 Classes                                │
│ ✅ Vantagens:                                                        │
│    • Simples de implementar                                          │
│    • Menos regras geradas                                            │
│    • Decisão automática (classe = round(saída))                     │
│    • Usa WangMendel (mais rápido)                                   │
│                                                                       │
│ ❌ Desvantagens:                                                     │
│    • Menos controle sobre as partições                               │
│    • Assume que classes são "ordenadas" (0 < 1 < 2)                │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ABORDAGEM 2: Três Saídas Binárias (One-vs-All)                     │
│ ✅ Vantagens:                                                        │
│    • Total controle sobre cada classificador                         │
│    • Pode ajustar partições diferentes por classe                    │
│    • Melhor para classes desbalanceadas                              │
│    • Dá "scores de confiança" para cada classe                      │
│    • Útil quando classes não são mutuamente exclusivas               │
│                                                                       │
│ ❌ Desvantagens:                                                     │
│    • Mais complexo de implementar                                    │
│    • 3x mais regras geradas                                          │
│    • Precisa decidir classe final (argmax)                           │
│    • Usa WangMendelFIS (precisa criar FIS primeiro)                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

🎯 QUANDO USAR CADA UMA:

• Use ABORDAGEM 1 quando:
  - Quer simplicidade e rapidez
  - Classes são bem separadas
  - Não precisa de scores de confiança
  - Dataset balanceado

• Use ABORDAGEM 2 quando:
  - Precisa de controle fino sobre cada classe
  - Classes desbalanceadas (ex: 90% classe A, 5% B, 5% C)
  - Quer scores de confiança
  - Pode ter exemplos pertencentes a múltiplas classes
  - Classificação hierárquica

📊 PARA O IRIS:
   Ambas funcionam bem! Recomendo ABORDAGEM 1 pela simplicidade.
   Use ABORDAGEM 2 se precisar explicar a "confiança" da predição.
    """)


def main():
    print("\n")
    print("*" * 70)
    print(" " * 15 + "WANG-MENDEL PARA CLASSIFICAÇÃO IRIS")
    print(" " * 20 + "Comparando Duas Abordagens")
    print("*" * 70)

    # Abordagem 1
    # wm1, sys1 = abordagem_1_saida_unica()

    input("\n\nPressione Enter para continuar para a Abordagem 2...")

    # Abordagem 2
    wm2, sys2 = abordagem_2_one_vs_all()

    if wm2 is not None:  # Se gerou regras
        input("\n\nPressione Enter para ver a comparação...")
        comparacao_abordagens()
    else:
        print("\n⚠️  Abordagem 2 não conseguiu gerar regras completas.")
        print("Para classificação multi-classe, recomendamos a Abordagem 1!")

    print("\n" + "=" * 70)
    print("FIM DO EXEMPLO")
    print("=" * 70)


if __name__ == "__main__":
    main()
