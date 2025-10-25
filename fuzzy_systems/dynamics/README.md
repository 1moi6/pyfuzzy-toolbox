# Módulo Dynamics - Sistemas Dinâmicos Fuzzy

## 📚 Visão Geral

O módulo `dynamics` implementa ferramentas para modelagem e simulação de sistemas dinâmicos com incerteza fuzzy:

1. **Solver de EDOs Fuzzy**: Resolve EDOs com condições iniciais e/ou parâmetros fuzzy
2. **Sistemas p-Fuzzy**: Sistemas dinâmicos onde a função de evolução é definida por regras fuzzy

---

## 🔬 Solver de EDOs Fuzzy (α-níveis)

### Método de α-Níveis

Propaga incerteza fuzzy através de EDOs usando o **princípio de extensão de Zadeh**.

**Algoritmo:**
1. Gera n α-níveis (0 a 1)
2. Para cada α, extrai intervalos [min, max] dos números fuzzy
3. Constrói grid de pontos usando produto cartesiano
4. Resolve EDO para cada ponto do grid (vetorizado + paralelo)
5. Extrai envelope (min/max) em cada instante de tempo
6. Retorna solução fuzzy como conjunto de envelopes por α-nível

### FuzzyNumber - Números Fuzzy

Integrado com `fuzzy_systems.core`:

```python
from fuzzy_systems.dynamics import FuzzyNumber

# Triangular
num1 = FuzzyNumber.triangular(center=10, spread=2)

# Gaussiano
num2 = FuzzyNumber.gaussian(mean=5, sigma=1)

# Trapezoidal
num3 = FuzzyNumber.trapezoidal(a=1, b=2, c=3, d=4)

# A partir de FuzzySet do core
from fuzzy_systems.core import FuzzySet
fs = FuzzySet(name="temp", mf_type="gaussian", params=(20, 3))
num4 = FuzzyNumber.from_fuzzy_set(fs, support=(11, 29))
```

**Métodos:**
- `alpha_cut(alpha)`: Extrai intervalo [min, max] para nível α
- `membership(x)`: Calcula grau de pertinência

### FuzzyODESolver

Resolve EDOs com CIs e/ou parâmetros fuzzy.

**Exemplo 1: Crescimento Exponencial (CI Fuzzy)**

```python
from fuzzy_systems.dynamics import FuzzyNumber, FuzzyODESolver
import numpy as np

# Define EDO: dy/dt = k*y
def growth(t, y, k):
    return k * y[0]

# Condição inicial fuzzy: y(0) ~ 10 ± 2
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Resolver
solver = FuzzyODESolver(
    ode_func=growth,
    t_span=(0, 5),
    y0_fuzzy=[y0],
    params={'k': 0.5},  # k crisp
    n_alpha_cuts=11,
    n_grid_points=5
)

sol = solver.solve()
sol.plot()
```

**Exemplo 2: Parâmetro Fuzzy**

```python
# Decaimento: dy/dt = -λ*y
def decay(t, y, lam):
    return -lam * y[0]

# Parâmetro fuzzy: λ ~ 0.3 ± 0.05
lambda_fuzzy = FuzzyNumber.triangular(center=0.3, spread=0.05)

solver = FuzzyODESolver(
    ode_func=decay,
    t_span=(0, 10),
    y0_fuzzy=[100.0],  # CI crisp
    params={'lam': lambda_fuzzy},
    n_alpha_cuts=11
)

sol = solver.solve()
```

**Exemplo 3: Sistema de EDOs (Lotka-Volterra)**

```python
# Sistema presa-predador
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

# CIs fuzzy
presas_0 = FuzzyNumber.triangular(center=40, spread=5)
predadores_0 = FuzzyNumber.triangular(center=9, spread=2)

solver = FuzzyODESolver(
    ode_func=lotka_volterra,
    t_span=(0, 20),
    y0_fuzzy=[presas_0, predadores_0],
    params={
        'alpha': 1.1,
        'beta': 0.4,
        'delta': 0.1,
        'gamma': 0.4
    },
    n_alpha_cuts=7,
    n_grid_points=3,  # 3^2 = 9 combinações de CIs
    var_names=['Presas', 'Predadores']
)

sol = solver.solve()

# Plota presas
sol.plot(var_idx=0)

# Plota predadores
sol.plot(var_idx=1)
```

### FuzzySolution - Resultado da Solução

```python
sol.t              # Tempos
sol.y_min          # Envelopes inferiores [n_alpha, n_vars, n_time]
sol.y_max          # Envelopes superiores [n_alpha, n_vars, n_time]
sol.alphas         # Níveis α
sol.var_names      # Nomes das variáveis

# Métodos
sol.get_alpha_level(0.5)      # Retorna (y_min, y_max) para α=0.5
sol.plot(var_idx=0)           # Plota com α-níveis coloridos
sol.to_dataframe(alpha=1.0)   # Converte para pandas DataFrame
sol.to_csv('results.csv')     # Exporta para CSV
```

### Parâmetros do Solver

```python
FuzzyODESolver(
    ode_func: Callable,              # EDO: dy/dt = f(t, y, **params)
    t_span: Tuple[float, float],     # (t0, tf)
    y0_fuzzy: List[FuzzyNumber|float], # CIs fuzzy ou crisp
    params: Dict = None,              # {nome: FuzzyNumber|float}
    n_alpha_cuts: int = 11,          # Número de α-níveis
    n_grid_points: int = 3,          # Pontos por dimensão no grid
    method: str = 'RK45',            # Método ODE: RK45, DOP853, Radau, etc
    t_eval: np.ndarray = None,       # Tempos específicos (None=automático)
    n_jobs: int = -1,                # Paralelização (-1=todos os cores)
    rtol: float = 1e-6,              # Tolerância relativa
    atol: float = 1e-9,              # Tolerância absoluta
    var_names: List[str] = None      # Nomes das variáveis
)
```

### Otimizações Implementadas

**Vetorização:**
```python
# Grid construction (vetorizado)
y0_meshgrid = np.meshgrid(*y0_points, indexing='ij')
y0_grid = np.stack([grid.flatten() for grid in y0_meshgrid], axis=1)

# Envelope extraction (vetorizado)
solutions_array = np.stack(valid_solutions, axis=0)
y_min = np.min(solutions_array, axis=0)
y_max = np.max(solutions_array, axis=0)
```

**Paralelização:**
```python
from joblib import Parallel, delayed

solutions = Parallel(n_jobs=-1)(
    delayed(solve_single_ode)(y0, params)
    for y0, params in zip(y0_grid, params_grid)
)
```

**Speedup**: ~4-8x em CPU de 8 cores

### Performance

**Exemplo (Crescimento Logístico):**
- 1 variável
- 11 α-níveis
- 5 grid points por dimensão
- 1 parâmetro fuzzy (5 pontos)
- **Total**: 11 × 5 × 5 = 275 EDOs resolvidas
- **Tempo**: ~2-3 segundos (com paralelização)

**Exemplo (Lotka-Volterra):**
- 2 variáveis
- 7 α-níveis
- 3×3 grid (9 combinações de CIs)
- **Total**: 7 × 9 = 63 EDOs (sistemas 2D)
- **Tempo**: ~1-2 segundos

### Dicas de Uso

**Escolha de `n_alpha_cuts`:**
- **Poucos (5-7)**: Mais rápido, menos suave
- **Médio (11-15)**: Balanceado (recomendado)
- **Muitos (20+)**: Mais suave, mais lento

**Escolha de `n_grid_points`:**
- **2**: Apenas extremos [min, max] - rápido mas pode perder informação
- **3**: Extremos + centro - bom balanço (recomendado)
- **5+**: Mais pontos, mais preciso, mais lento

**Dica**: Para sistemas de múltiplas EDOs, use `n_grid_points=3` (grid cresce exponencialmente!)

**Método ODE:**
- **RK45**: Padrão, bom para maioria dos casos
- **DOP853**: Mais preciso, para problemas suaves
- **Radau**: Para problemas stiff
- **LSODA**: Adapta automaticamente entre stiff/non-stiff

### Exportação de Dados

```python
# Para pandas DataFrame
df = sol.to_dataframe(alpha=1.0)  # Núcleo fuzzy
print(df.head())

# Para CSV (formato internacional)
sol.to_csv('solucao.csv')

# Para CSV (formato brasileiro/Excel)
sol.to_csv('solucao.csv', sep=';', decimal=',')

# DataFrame tem metadados
print(df.attrs['alpha_level'])      # 1.0
print(df.attrs['n_alpha_levels'])   # 11
print(df.attrs['var_names'])        # ['y0', 'y1']
```

---

## 🌊 Sistemas p-Fuzzy

Sistemas dinâmicos onde a **função de evolução** é definida por um **sistema de inferência fuzzy** (Mamdani ou Sugeno).

### Tipos

1. **PFuzzyDiscrete**: Sistemas discretos
   - `absolute`: x_{n+1} = x_n + f(x_n)
   - `relative`: x_{n+1} = x_n * f(x_n)

2. **PFuzzyContinuous**: Sistemas contínuos
   - `absolute`: dx/dt = f(x)
   - `relative`: dx/dt = x * f(x)

### Exemplo: Sistema Discreto

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.dynamics import PFuzzyDiscrete
from fuzzy_systems.inference.rules import FuzzyRule

# 1. Criar FIS para definir evolução
fis = MamdaniSystem()

# População
pop = fis.add_input('population', (0, 100))
pop.add_term('baixa', 'trapezoidal', (0, 0, 20, 40))
pop.add_term('media', 'triangular', (30, 50, 70))
pop.add_term('alta', 'trapezoidal', (60, 80, 100, 100))

# Taxa de crescimento
taxa = fis.add_output('growth_rate', (-10, 10))
taxa.add_term('negativa', 'triangular', (-10, -5, 0))
taxa.add_term('estavel', 'triangular', (-2, 0, 2))
taxa.add_term('positiva', 'triangular', (0, 5, 10))

# Regras
fis.rule_base.add_rule(FuzzyRule({'population': 'baixa'}, {'growth_rate': 'positiva'}))
fis.rule_base.add_rule(FuzzyRule({'population': 'media'}, {'growth_rate': 'estavel'}))
fis.rule_base.add_rule(FuzzyRule({'population': 'alta'}, {'growth_rate': 'negativa'}))

# 2. Criar sistema p-fuzzy
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['population'],
    dt=1.0
)

# 3. Simular
trajectory = pfuzzy.simulate(
    x0={'population': 10},
    n_steps=50
)

# 4. Visualizar
pfuzzy.plot_trajectory()
```

### Exemplo: Sistema Contínuo

```python
from fuzzy_systems.dynamics import PFuzzyContinuous

# FIS define a taxa de resfriamento
fis = criar_fis_temperatura()

# Sistema contínuo
pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['temperature'],
    method='rk4'  # ou 'euler'
)

# Simular
trajectory = pfuzzy.simulate(
    x0={'temperature': 80},
    t_span=(0, 10),
    dt=0.1
)

# Plota
pfuzzy.plot_trajectory()
```

### Exemplo: Sistema 2D (Presa-Predador Fuzzy)

```python
# FIS com 2 entradas (presas, predadores) e 2 saídas (taxas)
fis = criar_fis_predador_presa()

# Sistema p-fuzzy
pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['presas', 'predadores'],
    method='rk4'
)

# Simular
trajectory = pfuzzy.simulate(
    x0={'presas': 40, 'predadores': 9},
    t_span=(0, 20),
    dt=0.01
)

# Espaço de fase
pfuzzy.plot_phase_space('presas', 'predadores')
```

### Exportação

```python
# Para CSV
pfuzzy.to_csv('trajetoria.csv')

# Formato brasileiro
pfuzzy.to_csv('trajetoria.csv', sep=';', decimal=',')
```

### Validação de Domínio

Se o estado sair do domínio definido no FIS:
```
⚠️  AVISO: Variável 'population' = 105.234 está fora do domínio [0, 100].
    Simulação interrompida.
    Passo: 45/100
    Tempo: 4.5000
```

A trajetória é truncada no ponto onde saiu do domínio.

---

## 📁 Arquivos do Módulo

```
dynamics/
├── __init__.py
├── fuzzy_ode.py          # Solver de EDO Fuzzy
├── pfuzzy.py             # Sistemas p-Fuzzy
└── README.md             # Este arquivo
```

---

## 📚 Exemplos

Consulte a pasta `examples/03_dynamics/`:

- `example_pfuzzy_simple.py` - p-Fuzzy básico
- `example_pfuzzy_population.py` - Modelo de população
- `example_pfuzzy_predator_prey.py` - Lotka-Volterra fuzzy

**Testes:**
- `examples/tests/test_fuzzy_ode.py` - 4 testes completos de EDO Fuzzy

---

## 🎓 Referências

### EDO Fuzzy (α-níveis)
- Zadeh, L. A. (1975). "The concept of a linguistic variable and its application to approximate reasoning"
- Buckley, J. J., & Feuring, T. (2000). "Fuzzy differential equations"
- Bede, B., & Gal, S. G. (2005). "Generalizations of the differentiability of fuzzy-number-valued functions"

### Sistemas p-Fuzzy
- Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017). "A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics"
- Pedrycz, W., & Gomide, F. (2007). "Fuzzy Systems Engineering: Toward Human-Centric Computing"

### Métodos Numéricos
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary Differential Equations I: Nonstiff Problems"
- Dormand, J. R., & Prince, P. J. (1980). "A family of embedded Runge-Kutta formulae" (RK45)

---

## ⚠️ Limitações

### EDO Fuzzy
1. **Grid exponencial**: Para muitas variáveis/parâmetros fuzzy, grid cresce exponencialmente
2. **Métodos numéricos**: α-cortes são aproximados numericamente
3. **Memória**: Armazena todas as soluções em memória

### p-Fuzzy
1. **Domínio fixo**: Variáveis devem permanecer dentro do domínio do FIS
2. **Estabilidade**: Usuário deve garantir que as regras produzem sistema estável

---

## 💡 TODOs Futuros

- [ ] Grid adaptativo (mais pontos onde há mais variação)
- [ ] α-cortes analíticos para funções específicas
- [ ] Streaming de soluções (não armazenar tudo)
- [ ] Suporte a DAEs (equações algébrico-diferenciais)
- [ ] Análise de sensibilidade automática
- [ ] p-Fuzzy com eventos/descontinuidades

---

**Versão**: 1.1
**Data**: 2025-10-25
**Status**: ✅ Completo e testado
