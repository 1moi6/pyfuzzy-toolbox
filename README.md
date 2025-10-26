# pyfuzzy-toolbox

[![PyPI version](https://badge.fury.io/py/pyfuzzy-toolbox.svg)](https://badge.fury.io/py/pyfuzzy-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfuzzy-toolbox.svg)](https://pypi.org/project/pyfuzzy-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfuzzy-toolbox)](https://pepy.tech/project/pyfuzzy-toolbox)

**Fuzzy Systems** é uma biblioteca Python completa para Sistemas Fuzzy, desenvolvida com foco em aplicações didáticas e profissionais. Vai além da inferência básica, incluindo aprendizado, equações diferenciais fuzzy e sistemas p-fuzzy.

## 🚀 Quick Start

### 📓 Interactive Notebooks (Recommended for Learning)

Explore **11 Colab-ready notebooks** covering fundamentals, inference systems, and machine learning:

**👉 [Access Interactive Notebooks](../notebooks_colab/)**

Topics include:
- 🔰 Fundamentals: Membership functions, fuzzy sets, operators
- 🎛️ Inference Systems: Mamdani, Sugeno (order 0 and 1)
- 🧠 Learning: Wang-Mendel, ANFIS, PSO optimization

All notebooks can be opened directly in Google Colab with a single click!

---

### ⚡ Quick Code Example

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

## 💡 Exemplos de Uso

### Exemplo Básico: Controle de Ventilador

```python
import fuzzy_systems as fs

# Criar sistema
system = fs.MamdaniSystem()

# Adicionar variável de entrada
system.add_input('temperatura', (0, 40))
system.add_term('temperatura', 'fria', 'triangular', (0, 0, 20))
system.add_term('temperatura', 'morna', 'triangular', (10, 20, 30))
system.add_term('temperatura', 'quente', 'triangular', (20, 40, 40))

# Adicionar variável de saída
system.add_output('ventilador', (0, 100))
system.add_term('ventilador', 'lento', 'triangular', (0, 0, 50))
system.add_term('ventilador', 'medio', 'triangular', (25, 50, 75))
system.add_term('ventilador', 'rapido', 'triangular', (50, 100, 100))

# Adicionar regras (forma simples com listas)
system.add_rules([
    (['fria'], ['lento']),
    (['morna'], ['medio']),
    (['quente'], ['rapido'])
])

# Avaliar
resultado = system.evaluate({'temperatura': 25})
print(f"Velocidade: {resultado['ventilador']:.1f}%")
```

### Sistema Sugeno/TSK

```python
import fuzzy_systems as fs

# Sistema Sugeno com saídas funcionais
system = fs.SugenoSystem()

system.add_input('x', (0, 10))
system.add_term('x', 'baixo', 'triangular', (0, 0, 5))
system.add_term('x', 'alto', 'triangular', (5, 10, 10))

# Saída = função linear: a*x + b
system.add_output('y', order=1)  # Ordem 1 (linear)

# Regras com coeficientes
system.add_rules([
    (['baixo'], [2.0, 1.0]),   # y = 2*x + 1
    (['alto'], [0.5, 3.0])      # y = 0.5*x + 3
])

resultado = system.evaluate({'x': 7})
```

### Aprendizado com ANFIS

```python
import fuzzy_systems as fs
import numpy as np

# Dados de treinamento
X_train = np.random.uniform(0, 10, (100, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

# Criar e treinar ANFIS
anfis = fs.learning.ANFIS(n_inputs=2, n_terms=3)
anfis.fit(X_train, y_train, epochs=50)

# Predizer
X_test = np.array([[5.0, 3.0]])
y_pred = anfis.predict(X_test)
```

## 📚 Documentação e Exemplos

### Exemplos Organizados por Nível

```
examples/
├── 01_inference/      # ⭐ Iniciante - Sistemas básicos
├── 02_learning/       # ⭐⭐ Intermediário - Aprendizado
├── 03_dynamics/       # ⭐⭐⭐ Avançado - Sistemas dinâmicos
└── 04_complete/       # ⭐⭐⭐ Profissional - Aplicações completas
```

Consulte `examples/README.md` para guia completo.

## 🎯 Características Principais

- **Inferência Fuzzy**: Mamdani e Sugeno/TSK
- **Aprendizado**: ANFIS, Wang-Mendel, otimização com PSO/DE/GA
- **Sistemas Dinâmicos**: EDOs fuzzy, p-fuzzy
- **API Simples**: Menos código, mais produtividade
- **Bem Documentado**: 16 exemplos + notebooks interativos

## 📝 Citação

```bibtex
@software{pyfuzzy_toolbox,
  title = {pyfuzzy-toolbox: A Comprehensive Python Library for Fuzzy Systems},
  author = {Cecconello, Moiseis},
  year = {2025},
  url = {https://github.com/1moi6/pyfuzzy-toolbox},
  note = {Includes inference, learning, fuzzy differential equations, and p-fuzzy systems}
}
```

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

**PyPI**: https://pypi.org/project/pyfuzzy-toolbox/  
**GitHub**: https://github.com/1moi6/pyfuzzy-toolbox
