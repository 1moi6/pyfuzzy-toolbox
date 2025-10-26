# Notebooks Colab - pyfuzzy-toolbox

Este diretório contém versões Colab-ready dos notebooks, convertidos para usar a biblioteca **pyfuzzy-toolbox** diretamente do PyPI.

## 🎯 Diferenças das Versões Originais

### Versão Original
- Usa imports locais: `sys.path.insert(0, '/Users/...')`
- Requer código fonte local
- Não funciona no Google Colab

### Versão Colab (Este diretório) ✅
- Instala via PyPI: `!pip install pyfuzzy-toolbox`
- Funciona em qualquer ambiente (Colab, local, etc.)
- Badge "Open in Colab" em cada notebook
- Imports padronizados: `import fuzzy_systems as fs`

---

## 🗂️ Estrutura

```
notebooks_colab/
├── 01_fundamentals/        # Conceitos básicos de lógica fuzzy
├── 02_inference/           # Sistemas de inferência (Mamdani, Sugeno)
├── 03_learning/            # Aprendizado e otimização (Wang-Mendel, ANFIS, PSO)
└── 04_dynamics/            # Sistemas dinâmicos p-fuzzy (discretos e contínuos)
```

---

## 📚 01_fundamentals/ - Fundamentos de Lógica Fuzzy

### 01_membership_functions.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/01_membership_functions.ipynb)

**Conteúdo:**
- Funções de pertinência (triangular, trapezoidal, gaussiana, sigmoidal)
- Classes `FuzzySet` e `LinguisticVariable`
- Processo de fuzzificação
- Operadores fuzzy (AND, OR, NOT)
- Exercícios práticos

**Tempo estimado:** 45-60 minutos

---

### 02_thermal_comfort.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/02_thermal_comfort.ipynb)

**Conteúdo:**
- Modelagem de múltiplas variáveis linguísticas
- Sistema de conforto térmico (temperatura + umidade)
- Regras de inferência fuzzy
- Mapa 2D de conforto
- Exercícios personalizáveis

**Tempo estimado:** 40-50 minutos

---

## 🎛️ 02_inference/ - Sistemas de Inferência Fuzzy

### 01_mamdani_tipping.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/01_mamdani_tipping.ipynb)

**Conteúdo:**
- Sistema Mamdani completo usando `MamdaniSystem()`
- As 5 etapas do método Mamdani
- Sistema clássico de gorjeta (serviço + comida → gorjeta)
- Visualização do processo de inferência
- Superfície de controle 3D

**Tempo estimado:** 60-75 minutos

---

### 02_sugeno_zero_order.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/02_sugeno_zero_order.ipynb)

**Conteúdo:**
- Sistema Sugeno Ordem 0 (saídas constantes)
- Diferenças entre Mamdani e Sugeno
- Média ponderada como defuzzificação
- Exemplo didático: avaliação de desempenho
- Curva de resposta do sistema

**Tempo estimado:** 45-60 minutos

---

### 03_sugeno_first_order.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/03_sugeno_first_order.ipynb)

**Conteúdo:**
- Sistema Sugeno Ordem 1 (saídas lineares)
- Funções lineares: y = p₀ + p₁x₁ + p₂x₂
- Sistema com duas entradas
- Comparação Ordem 0 vs Ordem 1
- Superfície de controle 3D mais suave

**Tempo estimado:** 40-50 minutos

---

### 04_voting_prediction.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/04_voting_prediction.ipynb)

**Conteúdo:**
- Atividade prática completa
- Sistema com múltiplas entradas (renda + escolaridade)
- Predição de chance de voto
- Base de regras complexa
- Superfície 3D e mapa de contorno

**Tempo estimado:** 50-70 minutos

---

## 🧠 03_learning/ - Aprendizado e Otimização

### 01_wang_mendel_nonlinear.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/01_wang_mendel_nonlinear.ipynb)

**Conteúdo:**
- Método de Wang-Mendel para geração automática de regras
- Aproximação de função não-linear: f(x) = sin(x) + 0.1x
- Os 5 passos do algoritmo Wang-Mendel
- Particionamento fuzzy automático
- Resolução de conflitos entre regras
- Avaliação de desempenho (MSE, RMSE, R²)

**Tempo estimado:** 60-75 minutos

---

### 02_wang_mendel_linear.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/02_wang_mendel_linear.ipynb)

**Conteúdo:**
- Wang-Mendel aplicado a função linear: f(x) = -2x + 5
- Exemplo didático simples para entender particionamento
- Visualização de partições fuzzy
- Experimento: efeito do número de partições (3, 5, 7, 11)
- Trade-off precisão vs complexidade

**Tempo estimado:** 40-50 minutos

---

### 03_wang_mendel_iris.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/03_wang_mendel_iris.ipynb)

**Conteúdo:**
- Wang-Mendel para **classificação** (dataset Iris)
- 2 features: Petal Length e Petal Width
- 3 classes: setosa, versicolor, virginica
- Regras interpretáveis para classificação
- Matriz de confusão e métricas de classificação
- Comparação com KNN e Decision Tree

**Tempo estimado:** 50-65 minutos

---

### 04_anfis_iris.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/04_anfis_iris.ipynb)

**Conteúdo:**
- ANFIS: combinação de Lógica Fuzzy + Redes Neurais
- Classificação binária do dataset Iris (Setosa vs Não-Setosa)
- Aprendizado de funções de pertinência via backpropagation
- Regularização L2 para evitar overfitting
- Visualização de fronteira de decisão
- Comparação ANFIS vs Wang-Mendel

**Destaques:**
- ✅ **Refinamento automático** de MFs (diferente de Wang-Mendel)
- ✅ **Gradient descent** para otimizar parâmetros
- ✅ **Early stopping** baseado em validação
- ✅ Mantém interpretabilidade das regras

**Tempo estimado:** 60-75 minutos

---

### 05_pso_optimization.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/05_pso_optimization.ipynb)

**Conteúdo:**
- **PSO (Particle Swarm Optimization)**: Otimização por enxame
- Otimização de parâmetros fuzzy para função linear f(x) = -2x + 5
- Comportamento coletivo do enxame
- Memória pessoal vs conhecimento global
- Otimiza médias, sigmas e centroides das MFs

**Conceitos:**
- 🐝 **Enxame**: Colaboração entre partículas
- 🧠 **Memória pessoal**: Melhor posição de cada partícula
- 🌍 **Conhecimento global**: Melhor posição de todas
- ⚡ **Velocidade**: Movimento adaptativo no espaço

**Tempo estimado:** 50-65 minutos

---

## 🌊 04_dynamics/ - Sistemas Dinâmicos p-Fuzzy

### 01_pfuzzy_discrete_predator_prey.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/01_pfuzzy_discrete_predator_prey.ipynb)

**Conteúdo:**
- Sistema p-fuzzy discreto: $x_{n+1} = x_n + f(x_n)$
- Modelo predador-presa (Lotka-Volterra discreto)
- 16 regras fuzzy baseadas no livro de Barros & Bassanezi
- Espaço de fase e dinâmica temporal
- Múltiplas condições iniciais
- Exportação de resultados para CSV

**Conceitos:**
- 🔢 **Sistemas discretos**: Evolução por passos
- 🦊 **Dinâmica populacional**: Interação entre espécies
- 📊 **Espaço de fase**: Visualização de trajetórias
- 🎯 **Regras linguísticas**: "SE presas=altas E predadores=baixos ENTÃO..."

**Tempo estimado:** 50-65 minutos

---

### 02_pfuzzy_continuous_predator_prey.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/02_pfuzzy_continuous_predator_prey.ipynb)

**Conteúdo:**
- Sistema p-fuzzy contínuo: $\frac{dx}{dt} = f(x)$
- Modelo Lotka-Volterra fuzzy contínuo
- Integração numérica: Euler vs Runge-Kutta 4ª ordem (RK4)
- Campo vetorial (quiver plot)
- Ciclos oscilatórios predador-presa
- Comparação de métodos de integração

**Conceitos:**
- 📐 **EDOs Fuzzy**: Equações diferenciais com regras fuzzy
- ⚙️ **Integração numérica**: RK4 vs Euler
- 🌀 **Ciclos oscilatórios**: Comportamento periódico
- 🧭 **Campo vetorial**: Direção do fluxo no espaço de fase

**Destaques:**
- ✅ **Precisão RK4**: 4 avaliações por passo
- ✅ **Interpretabilidade**: Regras linguísticas ao invés de parâmetros
- ✅ **Flexibilidade**: Fácil incorporar conhecimento especialista

**Tempo estimado:** 60-75 minutos

---

## 🚀 Como Usar

### No Google Colab

1. Clique no badge "Open in Colab" de qualquer notebook
2. O notebook abrirá no Google Colab
3. Execute a célula de instalação: `!pip install pyfuzzy-toolbox`
4. Execute as demais células sequencialmente

### Localmente (Jupyter)

```bash
# Instalar dependências
pip install pyfuzzy-toolbox jupyter

# Executar Jupyter
jupyter notebook

# Abrir o notebook desejado
```

### Localmente (VS Code)

1. Instalar extensão Jupyter para VS Code
2. Instalar pyfuzzy-toolbox: `pip install pyfuzzy-toolbox`
3. Abrir notebook e executar células

---

## 📦 Biblioteca pyfuzzy-toolbox

**PyPI:** https://pypi.org/project/pyfuzzy-toolbox/
**GitHub:** https://github.com/1moi6/pyfuzzy-toolbox

### Instalação

```bash
# Básico
pip install pyfuzzy-toolbox

# Com machine learning (ANFIS, Wang-Mendel)
pip install pyfuzzy-toolbox[ml]

# Completo
pip install pyfuzzy-toolbox[all]
```

### Import

```python
import fuzzy_systems as fs
from fuzzy_systems.core import LinguisticVariable, FuzzySet
from fuzzy_systems import MamdaniSystem, SugenoSystem
from fuzzy_systems.learning import WangMendel, ANFIS, PSO
```

---

## ✨ Principais Alterações na Conversão

1. **Instalação via PyPI** ao invés de imports locais
2. **Colab badges** para abertura direta
3. **Metadata Colab** nos notebooks
4. **Imports atualizados** para usar `fuzzy_systems`
5. **Nomes em inglês** para maior alcance internacional
6. **Organização temática** ao invés de por aulas

---

## 📊 Comparação de Métodos

| Método | Notebook | Tipo | Vantagem Principal |
|--------|----------|------|-------------------|
| **Mamdani** | 02_inference/01 | Inferência | Interpretável, regras linguísticas |
| **Sugeno** | 02_inference/02-03 | Inferência | Saídas lineares, mais eficiente |
| **Wang-Mendel** | 03_learning/01-03 | Aprendizado | Gera regras automaticamente |
| **ANFIS** | 03_learning/04 | Neuro-Fuzzy | Refina MFs via backpropagation |
| **PSO** | 03_learning/05 | Metaheurística | Otimiza parâmetros sem gradientes |
| **p-Fuzzy Discreto** | 04_dynamics/01 | Dinâmica | Sistemas discretos com regras fuzzy |
| **p-Fuzzy Contínuo** | 04_dynamics/02 | Dinâmica | EDOs com regras fuzzy, integração RK4 |

---

## 📝 Licença

MIT License - veja [LICENSE](../LICENSE)

---

**Desenvolvido para o Minicurso de Lógica Fuzzy**
**Biblioteca:** pyfuzzy-toolbox v1.0.0
**Autor:** Moiseis Cecconello
