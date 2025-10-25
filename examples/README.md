# Exemplos - Fuzzy Systems

Exemplos organizados por categoria e complexidade.

---

## 📁 Estrutura

```
examples/
├── 01_inference/       # Sistemas de inferência (Mamdani, Sugeno)
├── 02_learning/        # Aprendizado (ANFIS, Wang-Mendel)
├── 03_dynamics/        # Sistemas dinâmicos (p-Fuzzy, EDO Fuzzy)
├── 04_complete/        # Exemplos end-to-end completos
└── tests/              # Testes e validações
```

---

## 🚀 Começando

### Para iniciantes

Comece com os exemplos em `01_inference/`:
1. `01_basic_mamdani.py` - Seu primeiro sistema Mamdani
2. `02_basic_sugeno.py` - Sistema Sugeno básico
3. `03_tipping_problem.py` - Problema clássico de gorjeta

### Para uso avançado

Explore `04_complete/` para aplicações completas:
- **Controle fuzzy**: Sistema de temperatura com visualizações
- **ANFIS**: Aproximação de funções não-lineares
- **Híbrido**: Combinando conhecimento + dados
- **EDO Fuzzy**: Modelagem epidemiológica com incerteza

---

## 📂 01_inference/ - Sistemas de Inferência

Exemplos essenciais de criação e uso de sistemas Mamdani e Sugeno.

### Conceitos Básicos
- `01_basic_mamdani.py` - Seu primeiro sistema Mamdani
- `02_basic_sugeno.py` - Seu primeiro sistema Sugeno
- `03_tipping_problem.py` - Problema clássico de gorjeta (referência da literatura)

### Ferramentas Avançadas
- `11_visualization.py` - Visualização de MFs, variáveis e superfícies
- `12_rules_import_export.py` - Exportar/importar regras (CSV, JSON, TXT)

**O que você vai aprender**:
- Criar sistemas de inferência Mamdani e Sugeno
- Definir variáveis linguísticas e funções de pertinência
- Criar regras fuzzy
- Avaliar sistemas com diferentes entradas
- Visualizar funções de pertinência e respostas
- Exportar e reutilizar bases de regras

**Nível**: Iniciante

**Próximo passo**: Veja `04_complete/` para aplicações end-to-end completas

---

## 📂 02_learning/ - Aprendizado

Algoritmos de aprendizado e extração de regras a partir de dados.

### Wang-Mendel (Extração de Regras)
- `13_wang_mendel.py` - Introdução ao algoritmo Wang-Mendel (regressão)
- `14_wang_mendel_iris.py` - Classificação multi-classe (dataset Iris)
- `wang_mendel_iris.ipynb` - Notebook interativo Wang-Mendel

### ANFIS (Adaptive Neuro-Fuzzy)
- `example_anfis.ipynb` - Tutorial completo ANFIS com visualizações

**O que você vai aprender**:
- Extrair regras fuzzy automaticamente de dados
- Usar Wang-Mendel para regressão e classificação
- Treinar redes neuro-fuzzy (ANFIS) com backpropagation
- Estabilidade de Lyapunov e regularização
- Aprendizado híbrido (LSE + gradiente descendente)

**Nível**: Intermediário a Avançado

**Pré-requisito**: Completar `01_inference/` primeiro

---

## 📂 03_dynamics/ - Sistemas Dinâmicos

Sistemas dinâmicos onde a evolução é definida por regras fuzzy (p-Fuzzy).

### p-Fuzzy Systems (Progressão Pedagógica)
1. `example_pfuzzy_simple.py` - Introdução: sistema com 1 variável
2. `example_pfuzzy_population.py` - Modelo populacional (discreto + contínuo)
3. `example_pfuzzy_predator_prey.py` - Lotka-Volterra (sistema acoplado)

**O que você vai aprender**:
- Sistemas dinâmicos governados por regras fuzzy
- Diferença entre modo discreto e contínuo
- Modo absoluto vs incremental
- Modelagem de interações ecológicas
- Sistemas acoplados com múltiplos FIS

**Aplicações**: Ecologia, dinâmica populacional, sistemas controlados por lógica fuzzy

**Nível**: Avançado

**Pré-requisito**: Dominar `01_inference/` primeiro

---

## 📂 04_complete/ - Exemplos End-to-End

Aplicações profissionais completas demonstrando as 4 funcionalidades principais do pacote.

### 1. Controle de Temperatura (Mamdani)
**Arquivo**: `01_temperatura_confortavel_mamdani.py`

Sistema completo de controle de ventilador baseado em temperatura e umidade.

**Inclui**:
- Criação de FIS do zero
- Definição de variáveis linguísticas
- Regras de controle
- Superfícies de resposta 3D/2D
- Visualizações completas

**Aplicação**: Controle de climatização, automação residencial

---

### 2. ANFIS - Aproximação de Função
**Arquivo**: `02_anfis_aproximacao_funcao.py`

Aproximação de função não-linear 2D com ANFIS.

**Inclui**:
- Geração de dados de treino/teste
- Configuração e treinamento de ANFIS
- Estabilidade de Lyapunov
- Análise de convergência
- Comparação real vs predito
- Salvar/carregar modelo

**Função alvo**: f(x,y) = sin(x) * cos(y) + 0.1*x*y

**Aplicação**: Modelagem de sistemas não-lineares, controle adaptativo

---

### 3. Sistema Híbrido (Conhecimento + Dados)
**Arquivo**: `03_hibrido_conhecimento_dados.py`

Integração completa: MamdaniSystem ↔ MamdaniLearning

**Inclui**:
- FIS inicial com conhecimento especialista
- Conversão para modelo otimizável
- Ajuste fino com dados observados
- Otimização gradiente + metaheurística
- Comparação antes/depois
- Exportação de FIS otimizado

**Problema**: Controle de irrigação agrícola

**Aplicação**: Sistemas onde há conhecimento prévio + dados de campo

---

### 4. EDO Fuzzy - Modelo Epidemiológico
**Arquivo**: `04_edo_fuzzy_epidemiologia.py`

Modelo SIR (Suscetíveis-Infectados-Recuperados) com incerteza fuzzy.

**Inclui**:
- Modelo SIR clássico
- Parâmetros fuzzy (β, γ)
- Propagação de incerteza via α-níveis
- Análise de cenários
- Envelopes fuzzy
- Exportação de resultados (CSV, DataFrame)

**Aplicação**: Epidemiologia, planejamento de saúde pública, análise de risco

---

## 📂 tests/ - Testes e Validações

Scripts de teste para validar funcionalidades.

- `test_fuzzy_ode.py` - Testes de EDO fuzzy (4 casos)
- `test_mamdani_learning.py` - Testes de Mamdani Learning
- `test_mamdani_hybrid.py` - Testes do modo híbrido
- `test_metaheuristics.py` - Testes de PSO, DE, GA
- `test_melhorias.py` - Validações de melhorias

**Uso**: Validar implementações, regressão testing

---

## 🎯 Guia de Escolha

### "Quero aprender sistemas fuzzy"
→ Comece em `01_inference/`, depois `04_complete/01_temperatura_...`

### "Preciso de um sistema de controle"
→ `04_complete/01_temperatura_confortavel_mamdani.py`

### "Quero aprender a partir de dados"
→ `02_learning/` → depois `04_complete/02_anfis_...`

### "Tenho conhecimento prévio + dados"
→ `04_complete/03_hibrido_conhecimento_dados.py`

### "Preciso modelar incerteza em EDOs"
→ `04_complete/04_edo_fuzzy_epidemiologia.py`

### "Quero sistemas dinâmicos com regras"
→ `03_dynamics/example_pfuzzy_*.py`

---

## 🔧 Executando os Exemplos

### Requisitos

```bash
pip install fuzzy-systems
pip install numpy scipy matplotlib pandas scikit-learn joblib
```

### Executar

```bash
# Exemplo básico
python examples/01_inference/01_basic_mamdani.py

# Exemplo completo
python examples/04_complete/01_temperatura_confortavel_mamdani.py

# Notebooks (requer Jupyter)
jupyter notebook examples/02_learning/example_anfis.ipynb
```

---

## 📊 Visão Geral dos Exemplos

| Pasta | Arquivos | Nível | Tempo | Foco |
|-------|----------|-------|-------|------|
| `01_inference/` | 5 | ⭐ Iniciante | 10-30 min | Conceitos básicos de lógica fuzzy |
| `02_learning/` | 4 | ⭐⭐ Intermediário | 30-60 min | Aprendizado a partir de dados |
| `03_dynamics/` | 3 | ⭐⭐⭐ Avançado | 30-60 min | Sistemas dinâmicos p-fuzzy |
| `04_complete/` | 4 | ⭐⭐⭐ Avançado | 60-120 min | Aplicações profissionais completas |
| **TOTAL** | **16** | - | - | **4 módulos principais** |

---

## 💡 Guia de Uso

### Para Iniciantes
1. Comece por `01_inference/` na ordem (01 → 02 → 03)
2. Pratique com `11_visualization.py` para ver seus sistemas
3. Quando dominar, vá para `04_complete/01_temperatura_*`

### Para Intermediários
1. Complete `02_learning/` (Wang-Mendel e ANFIS)
2. Explore notebooks para experimentação interativa
3. Veja `04_complete/02_anfis_*` e `03_hibrido_*`

### Para Avançados
1. Estude `03_dynamics/` na progressão (simple → population → predator_prey)
2. Aplique em `04_complete/04_edo_fuzzy_*`
3. Combine técnicas em problemas próprios

### Dicas Gerais
- 📓 **Notebooks** são ideais para aprendizado interativo
- 🐍 **Scripts .py** são melhores para integração em projetos
- 📁 Todos os exemplos salvam visualizações em `/tmp/`
- ⚠️ Execute exemplos do diretório raiz do projeto

---

## 📚 Documentação

Para mais informações, consulte:

- **README principal**: `../README.md`
- **Learning**: `../fuzzy_systems/learning/README.md`
- **Dynamics**: `../fuzzy_systems/dynamics/README.md`
- **Integração**: `../MAMDANI_LEARNING_INTEGRATION.md`

---

## 🤝 Contribuindo

Para adicionar novos exemplos:

1. Escolha a pasta apropriada (`01-04/`)
2. Nomeie claramente: `NN_descricao_do_exemplo.py`
3. Inclua docstring completa no topo
4. Adicione prints informativos
5. Salve visualizações em `/tmp/`
6. Documente no README apropriado

---

**Versão**: 1.0
**Data**: 2025-10-25
**Status**: ✅ Organizado e documentado
