# pyfuzzy-toolbox

[![PyPI version](https://badge.fury.io/py/pyfuzzy-toolbox.svg)](https://badge.fury.io/py/pyfuzzy-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfuzzy-toolbox.svg)](https://pypi.org/project/pyfuzzy-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfuzzy-toolbox)](https://pepy.tech/project/pyfuzzy-toolbox)

**Fuzzy Systems** é uma biblioteca Python completa para Sistemas Fuzzy, desenvolvida com foco em aplicações didáticas e profissionais. Vai além da inferência básica, incluindo aprendizado, equações diferenciais fuzzy e sistemas p-fuzzy.

## 🚀 Quick Start

```python
import fuzzy_systems as fs

# Create a Mamdani system in just 3 lines!
system = fs.MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_output('fan_speed', (0, 100))

# Add linguistic terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules
system.add_rules([
    (['cold'], ['slow']),
    (['hot'], ['fast'])
])

# Evaluate!
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
```

## Características

### ✅ Implementado (v1.0.0)

**Sistemas de Inferência Fuzzy:**
- ✅ **Sistemas Mamdani**: Inferência fuzzy clássica com defuzzificação
- ✅ **Sistemas Sugeno/TSK**: Inferência com saídas funcionais (ordem 0 e 1)
- ✅ **Funções de Pertinência**: Triangular, trapezoidal, gaussiana, sino, sigmoide, singleton
- ✅ **Operadores Fuzzy**: Múltiplas t-normas e s-normas (min, max, produto, etc.)
- ✅ **Defuzzificação**: Centroide, bisector, mean of maximum, e mais
- ✅ **API Ultra Simplificada**: 60-80% menos código que bibliotecas tradicionais

**Aprendizado e Otimização:**
- ✅ **ANFIS**: Adaptive Neuro-Fuzzy Inference System com estabilidade de Lyapunov
- ✅ **Wang-Mendel**: Geração automática de regras a partir de dados
- ✅ **MamdaniLearning**: Otimização de sistemas Mamdani com gradiente e metaheurísticas
- ✅ **Metaheuristics**: PSO, Differential Evolution, Genetic Algorithms
- ✅ **Import/Export de Regras**: Salvar e carregar regras em JSON, CSV, TXT

**Sistemas Dinâmicos:**
- ✅ **Fuzzy ODE Solver**: Resolver EDOs com incerteza fuzzy (α-level method)
- ✅ **p-Fuzzy Systems**: Sistemas dinâmicos discretos e contínuos

**Integração:**
- ✅ **MamdaniSystem ↔ MamdaniLearning**: Conversão bidirecional entre sistemas

**Exemplos e Documentação:**
- ✅ **16 exemplos** organizados por complexidade
- ✅ **Notebooks interativos** Jupyter
- ✅ **Documentação completa** com guias de aprendizado

## 📦 Instalação

### Do PyPI (Recomendado)

```bash
# Instalação básica
pip install pyfuzzy-toolbox

# Com dependências de machine learning (ANFIS, Wang-Mendel, etc.)
pip install pyfuzzy-toolbox[ml]

# Com dependências de desenvolvimento (testes, formatação, etc.)
pip install pyfuzzy-toolbox[dev]

# Instalação completa (todas as dependências)
pip install pyfuzzy-toolbox[all]
```

**Nota:** O nome do pacote no PyPI é `pyfuzzy-toolbox`, mas o import é `fuzzy_systems`:

```python
import fuzzy_systems as fs  # Nome do import
```

### Do código fonte (desenvolvimento)

```bash
git clone https://github.com/1moi6/pyfuzzy-toolbox.git
cd pyfuzzy-toolbox
pip install -e .          # Instalação editável
pip install -e .[dev]     # Com dependências de desenvolvimento
```

## Início Rápido

### Sistema Mamdani - Duas Formas de Criação

O **fuzzy_systems** oferece duas formas de criar sistemas Mamdani, cada uma com suas vantagens:

#### Forma 1: Criação Rápida (Prototipagem)

Ideal para criar sistemas rapidamente com menos código:

```python
import fuzzy_systems as fs

# Cria um sistema Mamdani para controle de temperatura
system = fs.create_mamdani_system(
    input_specs={
        'temperatura': ((0, 40), {
            'fria': ('triangular', (0, 0, 20)),
            'morna': ('triangular', (10, 20, 30)),
            'quente': ('triangular', (20, 40, 40))
        })
    },
    output_specs={
        'ventilador': ((0, 100), {
            'lento': ('triangular', (0, 0, 50)),
            'medio': ('triangular', (25, 50, 75)),
            'rapido': ('triangular', (50, 100, 100))
        })
    },
    rules=[
        ({'temperatura': 'fria'}, {'ventilador': 'lento'}, 'AND'),
        ({'temperatura': 'morna'}, {'ventilador': 'medio'}, 'AND'),
        ({'temperatura': 'quente'}, {'ventilador': 'rapido'}, 'AND'),
    ]
)

# Avalia o sistema (múltiplos formatos de entrada)
output = system.evaluate({'temperatura': 25})  # Dicionário
# ou: system.evaluate([25])                    # Lista
# ou: system.evaluate(25)                      # Valor direto (1 entrada)
# ou: system.evaluate(temperatura=25)          # Kwargs

print(f"Velocidade do ventilador: {output['ventilador']:.2f}%")
```

#### Forma 2: Criação Manual (Maior Controle)

Ideal para sistemas complexos com customizações avançadas:

```python
import fuzzy_systems as fs

# 1. Cria sistema com configurações customizadas
system = fs.MamdaniSystem(
    name="Sistema Manual",
    and_method=fs.TNorm.PRODUCT,      # Produto ao invés de min
    implication_method='product',       # Implicação de Larsen
    defuzzification_method='bisector'   # Bisector ao invés de centroid
)

# 2. Define entradas (forma simplificada)
temp_var = fs.LinguisticVariable('temperatura', (0, 40))
temp_var.add_term('fria', 'triangular', (0, 0, 20))      # Simplificado!
temp_var.add_term('quente', 'triangular', (20, 40, 40))
system.add_input(temp_var)

# 3. Define saídas
vent_var = fs.LinguisticVariable('ventilador', (0, 100))
vent_var.add_term('lento', 'triangular', (0, 0, 50))
vent_var.add_term('rapido', 'triangular', (50, 100, 100))
system.add_output(vent_var)

# 4. Adiciona regras (com pesos opcionais)
system.add_rule(fs.FuzzyRule(
    {'temperatura': 'fria'},
    {'ventilador': 'lento'},
    'AND',
    weight=0.8  # Peso da regra
))

# Avalia o sistema
output = system.evaluate(temperatura=25)
```

**💡 Dica:** Use a **forma rápida** para prototipagem e a **forma manual** quando precisar de controle fino sobre operadores, implicação, agregação ou defuzzificação.

### Múltiplos Formatos de Entrada

O método `evaluate()` aceita entradas em vários formatos para maior flexibilidade:

```python
# 1. Dicionário (mais legível)
output = system.evaluate({'temperatura': 25, 'umidade': 70})

# 2. Argumentos nomeados (kwargs)
output = system.evaluate(temperatura=25, umidade=70)

# 3. Lista/Tupla ordenada (útil para batch processing)
output = system.evaluate([25, 70])

# 4. Argumentos diretos
output = system.evaluate(25, 70)

# 5. Array NumPy
import numpy as np
inputs = np.array([25, 70])
output = system.evaluate(inputs)
```

**Veja exemplos detalhados em `examples/05_input_formats.py`**

### Múltiplos Formatos para Adicionar Regras

O sistema suporta várias formas de adicionar regras, desde a mais verbosa até a ultra simplificada:

```python
# 1. FuzzyRule completo (mais verboso, mais controle)
system.add_rule(fs.FuzzyRule(
    {'temperatura': 'fria', 'umidade': 'seca'},
    {'ventilador': 'lento'},
    'AND',
    weight=0.9
))

# 2. Dicionários diretos (sem criar FuzzyRule)
system.add_rule(
    {'temperatura': 'fria', 'umidade': 'seca'},
    {'ventilador': 'lento'}
)

# 3. Tuplas/Listas ordenadas ✨ RECOMENDADA!
system.add_rule(['fria', 'seca'], ['lento'])
system.add_rule(('quente', 'umida'), ('rapido',))

# 4. Múltiplas regras de uma vez ✨
system.add_rules([
    (['fria', 'seca'], ['lento']),
    (['fria', 'umida'], ['lento']),
    (['quente', 'seca'], ['rapido']),
    (['quente', 'umida'], ['rapido']),
])

# 5. Lista de dicionários com if/then
system.add_rules([
    {
        'if': {'temperatura': 'fria', 'umidade': 'seca'},
        'then': {'ventilador': 'lento'}
    },
    {
        'if': {'temperatura': 'quente', 'umidade': 'umida'},
        'then': {'ventilador': 'rapido'},
        'op': 'AND',
        'weight': 0.9
    }
])
```

**Redução de código: De 5 linhas para 1 linha por regra (80% menos código)!**

**Veja exemplos detalhados em `examples/09_rules_simplified.py`**

### Sistema Sugeno

O sistema Sugeno também suporta toda a API simplificada! ✨

#### Sugeno Ordem 0 (Consequentes Constantes)

```python
import fuzzy_systems as fs

# Sistema Sugeno com API simplificada
system = fs.SugenoSystem(order=0)

# Adiciona variáveis (simplificado!)
system.add_input('temperatura', (0, 40))
system.add_output('potencia', (0, 100))

# Adiciona termos (direto no sistema!)
system.add_term('temperatura', 'fria', 'triangular', (0, 0, 20))
system.add_term('temperatura', 'quente', 'triangular', (20, 40, 40))

# Adiciona regras com consequentes constantes (tuplas simplificadas!)
system.add_rules([
    (['fria'], 25.0),      # IF temp IS fria THEN potencia = 25
    (['quente'], 75.0)     # IF temp IS quente THEN potencia = 75
])

# Avalia (múltiplos formatos!)
output = system.evaluate(25)           # Valor direto
# ou: system.evaluate({'temperatura': 25})
# ou: system.evaluate(temperatura=25)

print(f"Potência: {output['potencia']:.2f}")
```

#### Sugeno Ordem 1 (Consequentes Lineares - TSK)

```python
# Sistema TSK de primeira ordem
system = fs.SugenoSystem(order=1)

system.add_input('x', (0, 10))
system.add_input('y', (0, 10))
system.add_output('z', (0, 100))

system.add_term('x', 'baixo', 'triangular', (0, 0, 5))
system.add_term('x', 'alto', 'triangular', (5, 10, 10))
system.add_term('y', 'baixo', 'triangular', (0, 0, 5))
system.add_term('y', 'alto', 'triangular', (5, 10, 10))

# Consequentes são funções lineares [c1, c2, ..., constante]
# Formato: z = c1*x + c2*y + constante
system.add_rules([
    (['baixo', 'baixo'], [2, 3, 5]),     # z = 2*x + 3*y + 5
    (['baixo', 'alto'],  [1, 5, 10]),    # z = 1*x + 5*y + 10
    (['alto', 'baixo'],  [5, 1, 8]),     # z = 5*x + 1*y + 8
    (['alto', 'alto'],   [3, 4, 15])     # z = 3*x + 4*y + 15
])

# Avalia
output = system.evaluate(2, 8)  # x=2, y=8
print(f"z = {output['z']:.2f}")
```

**Veja exemplos detalhados em `examples/10_sugeno_simplified.py`**

### Usando Componentes Individualmente

```python
import numpy as np
import fuzzy_systems as fs

# Funções de pertinência
x = np.linspace(0, 100, 1000)
mf_triangular = fs.triangular(x, (0, 50, 100))
mf_gaussian = fs.gaussian(x, (50, 15))

# Operadores fuzzy
a, b = 0.7, 0.5
and_result = fs.fuzzy_and_min(a, b)  # 0.5
or_result = fs.fuzzy_or_max(a, b)    # 0.7
not_result = fs.fuzzy_not(a)          # 0.3

# Defuzzificação
aggregated_mf = np.maximum(mf_triangular * 0.6, mf_gaussian * 0.4)
crisp_output = fs.centroid(x, aggregated_mf)
```

### Adicionando Termos de Forma Simplificada

Você pode adicionar termos fuzzy de forma ainda mais direta:

```python
# ✨ Forma Nova (Recomendada - Limpa e Direta)
var = fs.LinguisticVariable('temperatura', (0, 40))
var.add_term('baixa', 'triangular', (0, 0, 20))
var.add_term('media', 'gaussian', (20, 5))
var.add_term('alta', 'trapezoidal', (15, 25, 35, 40))

# Forma Antiga (ainda funciona)
var.add_term(fs.FuzzySet('baixa', 'triangular', (0, 0, 20)))
```

**Veja mais em `examples/06_add_term_simplified.py`**

### Visualização de Sistemas Fuzzy

A biblioteca inclui funcionalidades completas de visualização! ✨

#### Visualizar Variável Linguística

```python
import fuzzy_systems as fs

# Cria variável
temp = fs.LinguisticVariable('temperatura', (0, 40))
temp.add_term('fria', 'triangular', (0, 0, 20))
temp.add_term('morna', 'triangular', (15, 25, 35))
temp.add_term('quente', 'triangular', (30, 40, 40))

# Plot simples
temp.plot()

# Plot customizado
temp.plot(
    colors=['blue', 'green', 'red'],
    linewidth=3,
    figsize=(12, 6),
    title='Temperatura - Funções de Pertinência'
)
```

#### Visualizar Sistema Completo

```python
# Plota todas variáveis (entradas e saídas)
system.plot_variables()

# Plota apenas entradas
system.plot_variables('input')

# Plota apenas saídas
system.plot_variables('output')
```

#### Visualizar Resposta do Sistema (2D)

```python
# Para sistemas com 1 entrada e 1 saída
system.plot_output('temperatura', 'ventilador')

# Para sistemas com múltiplas entradas,
# as outras são fixadas no ponto médio
system.plot_output('temperatura', 'ventilador', num_points=200)
```

**Veja exemplos completos em `examples/11_visualization.py`**

## Estrutura do Projeto

```
fuzzy_systems/
├── fuzzy_systems/              # Pacote principal
│   ├── __init__.py            # API pública unificada
│   │
│   ├── core/                  # Componentes fundamentais
│   │   ├── __init__.py
│   │   ├── membership.py      # Funções de pertinência
│   │   ├── operators.py       # Operadores fuzzy (t-normas, s-normas)
│   │   ├── fuzzification.py   # FuzzySet, LinguisticVariable
│   │   └── defuzzification.py # Métodos de defuzzificação
│   │
│   ├── inference/             # Sistemas de inferência
│   │   ├── __init__.py
│   │   ├── rules.py          # FuzzyRule, RuleBase, Engines
│   │   └── systems.py        # MamdaniSystem, SugenoSystem
│   │
│   ├── learning/              # Aprendizado e otimização (em desenvolvimento)
│   │   ├── __init__.py
│   │   ├── anfs.py          # ANFIS (TODO)
│   │   ├── wang_mendel.py    # Wang-Mendel (TODO)
│   │   ├── genetic.py        # Algoritmos genéticos (TODO)
│   │   └── pso.py            # Particle Swarm (TODO)
│   │
│   ├── dynamic/               # Sistemas dinâmicos (em desenvolvimento)
│   │   ├── __init__.py
│   │   ├── ode.py            # EDOs fuzzy (TODO)
│   │   ├── pfuzzy.py         # Sistemas p-fuzzy (TODO)
│   │   └── control.py        # Controladores fuzzy (TODO)
│   │
│   └── utils/                 # Utilitários (em desenvolvimento)
│       ├── __init__.py
│       ├── visualization.py   # Gráficos e plots (TODO)
│       ├── io.py             # Import/Export (JSON, XML, FIS) (TODO)
│       └── metrics.py        # Métricas de performance (TODO)
│
├── examples/                  # 10 exemplos completos funcionando
├── tests/                     # Testes unitários
├── setup.py                   # Instalação
└── README.md                  # Este arquivo
```

### Arquitetura Modular

A biblioteca está organizada em módulos independentes para facilitar manutenção e expansão:

- **`core/`**: Componentes básicos reutilizáveis (sem dependências internas)
- **`inference/`**: Sistemas de inferência completos (depende apenas de `core/`)
- **`learning/`**: Algoritmos de aprendizado (usa `inference/` para treinar sistemas)
- **`dynamic/`**: Sistemas dinâmicos e controladores (usa `inference/` e opcionalmente `learning/`)
- **`utils/`**: Ferramentas compartilhadas por todos os módulos

## Exemplos Disponíveis

Verifique a pasta `examples/` para exemplos completos:

- `01_basic_mamdani.py` - Sistema Mamdani básico
- `02_basic_sugeno.py` - Sistema Sugeno básico (ordem 0 e 1)
- `03_tipping_problem.py` - Problema clássico de gorjeta
- `04_mamdani_two_ways.py` - **Comparação das duas formas de criar Mamdani**
- `05_input_formats.py` - **Múltiplos formatos de entrada para evaluate()**
- `06_add_term_simplified.py` - **Forma simplificada de adicionar termos**
- `07_system_add_term.py` - **Adicionar termos diretamente no sistema**
- `08_ultra_simplified.py` - **API ultra simplificada (30% menos código)**
- `09_rules_simplified.py` - **Cinco formas de adicionar regras (80% redução)**
- `10_sugeno_simplified.py` - **API simplificada para Sugeno (ordem 0 e 1)**
- `11_visualization.py` - **Visualização de variáveis e sistemas** ✨ NOVO!

### Quando Usar Cada Forma?

#### 📦 Use Forma Rápida (`create_mamdani_system`) quando:
- ✓ Prototipagem rápida
- ✓ Testes e experimentos
- ✓ Sistemas simples com configurações padrão
- ✓ Exemplos didáticos
- ✓ Menos código é prioridade

#### 🔧 Use Forma Manual (`MamdaniSystem` + componentes) quando:
- ✓ Precisa de controle fino sobre operadores (AND, OR, NOT)
- ✓ Quer customizar implicação/agregação/defuzzificação
- ✓ Está construindo sistemas complexos
- ✓ Precisa modificar o sistema dinamicamente
- ✓ Quer adicionar/remover regras em tempo de execução
- ✓ Está integrando com outros componentes
- ✓ Quer usar pesos diferentes nas regras
- ✓ Precisa entender a estrutura interna do sistema

**Veja o exemplo completo em `examples/04_mamdani_two_ways.py`**

## Documentação

### Funções de Pertinência

```python
# Triangular
fs.triangular(x, (a, b, c))

# Trapezoidal
fs.trapezoidal(x, (a, b, c, d))

# Gaussiana
fs.gaussian(x, (mean, sigma))

# Sino Generalizado
fs.generalized_bell(x, (a, b, c))

# Sigmoide
fs.sigmoid(x, (a, c))

# Singleton
fs.singleton(x, value)
```

### Operadores

#### T-normas (AND)
- `fuzzy_and_min` - Mínimo (Zadeh)
- `fuzzy_and_product` - Produto algébrico
- `fuzzy_and_lukasiewicz` - Łukasiewicz
- `fuzzy_and_drastic` - Drástico
- `fuzzy_and_hamacher` - Hamacher

#### S-normas (OR)
- `fuzzy_or_max` - Máximo (Zadeh)
- `fuzzy_or_probabilistic` - OR probabilístico
- `fuzzy_or_bounded` - Soma limitada
- `fuzzy_or_drastic` - Drástico
- `fuzzy_or_hamacher` - Hamacher

### Métodos de Defuzzificação

- `centroid` - Centro de gravidade (COG)
- `bisector` - Bisector
- `mean_of_maximum` - Média dos máximos (MOM)
- `smallest_of_maximum` - Menor máximo (SOM)
- `largest_of_maximum` - Maior máximo (LOM)
- `weighted_average` - Média ponderada (Sugeno)

## Desenvolvimento

### Executar Testes

```bash
pytest tests/
```

### Formatação de Código

```bash
black fuzzy_systems/
```

### Verificação de Tipos

```bash
mypy fuzzy_systems/
```

## Roadmap

### Inferência Fuzzy
- [x] Sistema Mamdani
- [x] Sistema Sugeno/TSK (ordem 0 e 1)
- [x] Funções de pertinência principais
- [x] Operadores fuzzy (múltiplas t-normas e s-normas)
- [x] Métodos de defuzzificação
- [x] API ultra simplificada

### Aprendizado e Otimização
- [ ] ANFIS (Adaptive Neuro-Fuzzy Inference System)
- [ ] Wang-Mendel (geração automática de regras a partir de dados)
- [ ] Otimização de base de regras (GA, PSO, ACO)
- [ ] Aprendizado de parâmetros de funções de pertinência

### Sistemas Avançados
- [ ] Equações Diferenciais Fuzzy (EDOs fuzzy)
- [ ] Sistemas p-Fuzzy
- [ ] Fuzzy Type-2 (sistemas fuzzy de tipo 2)

### Ferramentas
- [ ] Visualização interativa (surface plots, membership functions)
- [ ] Exportação/importação de sistemas (JSON, XML, FIS)
- [ ] Compatibilidade com MATLAB Fuzzy Toolbox
- [ ] Interface gráfica (opcional)

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

MIT License - veja o arquivo `LICENSE` para detalhes.

## Citação

Se você usar este toolbox em trabalhos acadêmicos, por favor cite:

```bibtex
@software{pyfuzzy_toolbox,
  title = {pyfuzzy-toolbox: A Comprehensive Python Library for Fuzzy Systems},
  author = {Fuzzy Systems Contributors},
  year = {2024},
  url = {https://github.com/1moi6/pyfuzzy-toolbox},
  note = {Includes inference, learning, fuzzy differential equations, and p-fuzzy systems}
}
```

## Autores

Desenvolvido para o Minicurso de Sistemas Fuzzy.

## Suporte

Para dúvidas, problemas ou sugestões:
- Abra uma issue no GitHub
- Consulte a documentação
- Entre em contato com os autores
