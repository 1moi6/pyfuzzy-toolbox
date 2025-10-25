# Módulo Learning - Aprendizado e Otimização Fuzzy

## 📚 Visão Geral

O módulo `learning` implementa algoritmos de aprendizado e otimização para sistemas fuzzy:

- **ANFIS** (Adaptive Neuro-Fuzzy Inference System)
- **Wang-Mendel** (Extração de regras a partir de dados)
- **Mamdani Learning** (Neuro-fuzzy com gradiente e metaheurísticas)
- **Metaheurísticas** (PSO, DE, GA)

---

## 🧠 ANFIS - Adaptive Neuro-Fuzzy Inference System

### Características

- **Arquitetura híbrida**: TSK + aprendizado
- **Algoritmo de treinamento**: LSE (consequentes) + Gradiente (antecedentes)
- **Estabilidade de Lyapunov**: Taxa de aprendizado adaptativa
- **Regularização**: L1/L2 apenas nas larguras (sigmas), não nos centros
- **Funções de pertinência**: Gaussiana, Bell Generalizada, Sigmoide
- **Gradientes analíticos**: Implementados para todas as MFs
- **Métricas**: RMSE, MAE, R², MAPE, learning_rate

### Uso Básico

```python
from fuzzy_systems.learning import ANFIS
import numpy as np

# Dados de treinamento
X_train = np.random.uniform(0, 10, (100, 2))
y_train = X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3

# Criar ANFIS
anfis = ANFIS(
    n_inputs=2,
    n_rules=9,
    mf_type='gaussian'
)

# Treinar
history = anfis.fit(
    X_train, y_train,
    epochs=100,
    learning_rate=0.01,
    verbose=True
)

# Predizer
y_pred = anfis.predict(X_test)

# Avaliar
score = anfis.score(X_test, y_test)  # R²
```

### Estabilidade de Lyapunov

Taxa de aprendizado adaptativa para garantir convergência:

```
η_adaptativo = min(1.99 / ||∇E||², η_max)
```

Garante que a função de energia (erro) sempre decresce.

**Referência**: Wang, L. X., & Mendel, J. M. (1992). "Fuzzy basis functions, universal approximation, and orthogonal least-squares learning"

### Regularização Simplificada

- **L1/L2 aplicada APENAS nas larguras (sigmas)**
- **Centros NÃO são regularizados** (decisão baseada em teoria fuzzy)

```python
anfis = ANFIS(
    n_inputs=2,
    n_rules=9,
    regularization='l2',
    lambda_reg=0.01  # Apenas para sigmas
)
```

### Gradientes Analíticos

Implementados para todas as funções de pertinência:

- **Gaussiana**: `∂μ/∂mean`, `∂μ/∂sigma`
- **Bell Generalizada**: `∂μ/∂a`, `∂μ/∂b`, `∂μ/∂c`
- **Sigmoide**: `∂μ/∂a`, `∂μ/∂c`

Permite treinamento rápido e estável.

### Salvar/Carregar Modelo

```python
# Salvar
anfis.save('modelo_anfis.npz')

# Carregar
anfis_loaded = ANFIS.load('modelo_anfis.npz')
```

---

## 📊 Wang-Mendel - Extração de Regras

### Algoritmo

1. **Fuzzificação**: Particiona domínio das variáveis
2. **Geração de regras**: Uma regra por amostra
3. **Resolução de conflitos**: Regra com maior grau vence
4. **Base de regras**: Conjunto final de regras

### Uso

```python
from fuzzy_systems.learning import WangMendelRuleExtractor

# Criar extrator
wm = WangMendelRuleExtractor(
    n_mfs_per_input=[5, 5],  # 5 MFs para cada entrada
    mf_type='triangular'
)

# Extrair regras
wm.extract_rules(X_train, y_train)

# Converter para FIS
fis = wm.to_mamdani_system(
    input_names=['temperatura', 'umidade'],
    output_name='ventilador'
)

# Usar
resultado = fis.evaluate(temperatura=25, umidade=60)
```

---

## 🎯 Mamdani Learning - Neuro-Fuzzy

### Características

- **Arquitetura**: 4 camadas (Fuzzificação → Regras → Defuzzificação → Saída)
- **MFs de entrada**: Gaussianas (aprendíveis)
- **MFs de saída**: Singletons (centroides aprendíveis)
- **Aprendizado**: Gradiente (batch, online, mini-batch) + Metaheurísticas
- **Defuzzificação**: COG ou COS
- **Otimização**: Caching de ativações para eficiência

### Aprendizado por Gradiente

```python
from fuzzy_systems.learning.mamdani import MamdaniLearning

# Criar
mamdani = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=3,
    defuzz_method='cog'
)

# Treinar
mamdani.fit(
    X_train, y_train,
    epochs=100,
    learning_rate=0.01,
    batch_size=32,  # mini-batch
    mode='batch'    # ou 'online', 'mini-batch'
)

# Predizer
y_pred = mamdani.predict(X_test)
```

### Otimização Metaheurística

Três estratégias:

1. **`consequents_only`**: Otimiza apenas índices dos consequentes (rápido)
2. **`output_only`**: Otimiza apenas centroides de saída
3. **`hybrid`**: Otimiza consequentes + centroides simultaneamente

```python
# PSO - Otimizar apenas consequentes (mais rápido)
mamdani.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    optimize_params='consequents_only',
    n_particles=30,
    n_iterations=50
)

# Differential Evolution - Modo híbrido
mamdani.fit_metaheuristic(
    X_train, y_train,
    optimizer='de',
    optimize_params='hybrid',
    n_particles=30,
    n_iterations=100
)

# Genetic Algorithm
mamdani.fit_metaheuristic(
    X_train, y_train,
    optimizer='ga',
    optimize_params='output_only',
    n_particles=50,
    n_iterations=100
)
```

### Modo Híbrido (Hybrid)

Otimiza **consequentes** + **centroides** simultaneamente com caching parcial:

**Características:**
- Pré-computa ativações das regras (cache)
- Otimiza mapeamento consequentes + valores dos centroides
- ~2-3x mais lento que `consequents_only`
- ~2x mais rápido que `output_only`
- Melhor balanço flexibilidade/performance

**Funcionamento:**
```python
# Vetor de parâmetros: [consequent_indices, centroids]
# Exemplo: [2, 1, 0, 1, 2, ..., 10.5, 50.2, 89.7]
#          └─────regras────────┘  └──centroides──┘
```

### Caching de Ativações

Quando os conjuntos de entrada são fixos (não mudam durante otimização):

```python
# Pré-computa ativações UMA VEZ
membership_values = self._fuzzify_inputs(X)
firing_strengths = self._fire_rules(membership_values)
self._cached_activations = firing_strengths

# Reutiliza cache (não recalcula fuzzificação)
predictions = self._defuzzify_cog(
    self._cached_activations,  # CACHE!
    consequent_indices
)
```

**Speedup**: ~10-100x em otimização metaheurística

### Integração com MamdaniSystem

**Learning → FIS:**
```python
# Treinar modelo
mamdani.fit(X_train, y_train, epochs=100)

# Exportar como FIS
fis = mamdani.to_mamdani_system(
    input_names=['temperatura', 'umidade'],
    output_name='ventilador'
)

# Usar FIS
resultado = fis.evaluate(temperatura=25, umidade=60)
```

**FIS → Learning:**
```python
# Importar FIS existente
fis = criar_fis_manual()  # Com MFs gaussianas

# Converter para MamdaniLearning
mamdani = MamdaniLearning.from_mamdani_system(fis)

# Otimizar com dados
mamdani.fit(X_train, y_train, epochs=50)

# Exportar FIS otimizado
fis_otimizado = mamdani.to_mamdani_system(...)
```

**Consulte**: `MAMDANI_LEARNING_INTEGRATION.md` na raiz para exemplos completos

---

## ⚙️ Metaheurísticas

### PSO - Particle Swarm Optimization

```python
from fuzzy_systems.learning.metaheuristics import PSO

pso = PSO(
    n_particles=30,
    n_iterations=100,
    w=0.7,         # Inércia
    c1=1.5,        # Cognitivo
    c2=1.5,        # Social
    w_decay=0.99   # Decaimento de inércia
)

best_params, best_fitness, history = pso.optimize(
    objective_func,
    bounds,
    minimize=True
)
```

### DE - Differential Evolution

```python
from fuzzy_systems.learning.metaheuristics import DE

de = DE(
    pop_size=30,
    max_iter=100,
    F=0.8,          # Fator de mutação
    CR=0.9,         # Crossover
    strategy='best1' # ou 'rand1', 'rand2', 'best2'
)

best_params, best_fitness, history = de.optimize(
    objective_func,
    bounds,
    minimize=True
)
```

### GA - Genetic Algorithm

```python
from fuzzy_systems.learning.metaheuristics import GA

ga = GA(
    pop_size=50,
    max_gen=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism_size=5  # Preserva os 5 melhores
)

best_params, best_fitness, history = ga.optimize(
    objective_func,
    bounds,
    minimize=True
)
```

---

## 📁 Arquivos do Módulo

```
learning/
├── __init__.py
├── anfis.py              # ANFIS implementação
├── wang_mendel.py        # Extração de regras
├── mamdani.py            # Mamdani Learning
├── metaheuristics.py     # PSO, DE, GA
└── README.md             # Este arquivo
```

---

## 📚 Exemplos

Consulte a pasta `examples/02_learning/`:

- `15_anfis_exemplo.py` - ANFIS básico
- `13_wang_mendel.py` - Wang-Mendel
- `14_wang_mendel_iris.py` - Wang-Mendel com Iris dataset
- `example_anfis.ipynb` - Notebook ANFIS
- `wang_mendel_iris.ipynb` - Notebook Wang-Mendel

**Testes:**
- `examples/tests/test_mamdani_learning.py`
- `examples/tests/test_mamdani_hybrid.py`
- `examples/tests/test_metaheuristics.py`

---

## 🎓 Referências

### ANFIS
- Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system"
- Wang, L. X., & Mendel, J. M. (1992). "Fuzzy basis functions, universal approximation, and orthogonal least-squares learning"

### Wang-Mendel
- Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by learning from examples"

### Metaheurísticas
- Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization" (PSO)
- Storn, R., & Price, K. (1997). "Differential evolution" (DE)
- Holland, J. H. (1975). "Adaptation in natural and artificial systems" (GA)

### Estabilidade
- Lyapunov, A. M. (1892). "The general problem of the stability of motion"
- Slotine, J. J. E., & Li, W. (1991). "Applied Nonlinear Control"

---

**Versão**: 1.1
**Data**: 2025-10-25
**Status**: ✅ Completo e testado
