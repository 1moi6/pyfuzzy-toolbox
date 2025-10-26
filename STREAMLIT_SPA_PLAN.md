# Plano de Implementação: Streamlit SPA para Fuzzy Systems

**Versão:** 1.0
**Data:** 2025-10-25
**Biblioteca:** pyfuzzy-toolbox

---

## 📋 Visão Geral

Este documento descreve o plano de implementação de uma interface Streamlit SPA (Single Page Application) para a biblioteca **pyfuzzy-toolbox**. A interface permitirá aos usuários criar, configurar, testar e exportar sistemas fuzzy através de uma interface gráfica intuitiva, mantendo a opção de usar código Python puro.

### Objetivos

1. **Interface Visual Intuitiva**: Criar sistemas fuzzy sem escrever código
2. **Arquitetura SPA**: Navegação fluida usando `st.session_state` (sem páginas separadas)
3. **Modular e Extensível**: Componentes reutilizáveis e bem organizados
4. **Integração com pyfuzzy-toolbox**: Uso direto da biblioteca instalada via PyPI
5. **Export/Import**: Salvar e carregar configurações de sistemas

---

## 🏗️ Arquitetura SPA

### Padrão de Navegação (baseado em app/main.py existente)

```python
import streamlit as st

# Importar módulos
from modules import home, designer, simulator, exporter

# Inicializar session_state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Navegação SPA
def navigate_to(page_name):
    st.session_state['page'] = page_name
    st.rerun()

# Renderizar página atual
current_page = st.session_state['page']

if current_page == 'home':
    home.run()
elif current_page == 'designer':
    designer.run()
elif current_page == 'simulator':
    simulator.run()
elif current_page == 'exporter':
    exporter.run()
```

### Estrutura de Pastas

```
streamlit_app/
├── main.py                      # Ponto de entrada SPA
├── config.py                    # Configurações globais
├── requirements.txt             # Dependências (pyfuzzy-toolbox, streamlit)
├── modules/                     # Módulos de página
│   ├── __init__.py
│   ├── home.py                  # Página inicial com tutorial
│   ├── designer.py              # Designer de sistemas fuzzy
│   ├── simulator.py             # Simulador/testador
│   └── exporter.py              # Exportar/importar sistemas
├── components/                  # Componentes reutilizáveis
│   ├── __init__.py
│   ├── sidebar.py               # Barra lateral de navegação
│   ├── variable_creator.py      # Criar variáveis linguísticas
│   ├── term_editor.py           # Editar termos fuzzy
│   ├── rule_builder.py          # Construtor visual de regras
│   ├── plotter.py               # Visualizações interativas
│   └── code_generator.py        # Gerar código Python equivalente
├── utils/                       # Utilidades
│   ├── __init__.py
│   ├── session_manager.py       # Gerenciar session_state
│   ├── system_converter.py      # Converter entre formatos
│   └── validators.py            # Validações de entrada
└── assets/                      # Recursos estáticos
    ├── logo.png
    └── examples/                # Sistemas de exemplo
        ├── temperature_control.json
        ├── tipping_system.json
        └── anfis_example.json
```

---

## 🎨 Módulos Principais

### 1. home.py - Página Inicial

**Responsabilidades:**
- Apresentar a biblioteca pyfuzzy-toolbox
- Tutorial interativo rápido
- Links para documentação e GitHub
- Sistemas de exemplo pré-carregados

**Interface:**
```python
def run():
    st.title("🌟 pyfuzzy-toolbox - Interface Visual")

    # Seção de boas-vindas
    st.markdown("""
    Bem-vindo à interface visual da **pyfuzzy-toolbox**!

    Esta aplicação permite criar e testar sistemas fuzzy de forma visual,
    sem escrever código. Você também pode exportar o código Python equivalente
    para uso em seus projetos.
    """)

    # Quick start
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎨 Designer", use_container_width=True):
            navigate_to('designer')
    with col2:
        if st.button("🧪 Simulador", use_container_width=True):
            navigate_to('simulator')
    with col3:
        if st.button("💾 Exportar", use_container_width=True):
            navigate_to('exporter')

    # Exemplos
    st.subheader("📚 Exemplos Prontos")
    example = st.selectbox("Carregar exemplo:",
                          ["Controle de Temperatura", "Sistema de Gorjeta", "ANFIS"])
    if st.button("Carregar"):
        load_example(example)
        navigate_to('designer')
```

---

### 2. designer.py - Designer de Sistemas Fuzzy

**Responsabilidades:**
- Criar/editar variáveis de entrada e saída
- Definir termos linguísticos
- Construir regras fuzzy
- Visualizar funções de pertinência em tempo real

**Workflow:**

```
1. Escolher tipo de sistema (Mamdani/Sugeno/ANFIS)
2. Adicionar variáveis de entrada
   └─> Para cada variável:
       - Definir nome e universo de discurso
       - Adicionar termos linguísticos
       - Escolher função de pertinência e parâmetros
       - Visualizar em tempo real
3. Adicionar variável(is) de saída
   └─> Similar às entradas
4. Construir regras
   └─> Interface drag-and-drop ou seleção
5. Pré-visualizar sistema completo
```

**Interface (pseudo-código):**

```python
def run():
    st.title("🎨 Designer de Sistemas Fuzzy")

    # Sidebar: Tipo de sistema
    with st.sidebar:
        system_type = st.selectbox("Tipo de Sistema:",
                                   ["Mamdani", "Sugeno", "ANFIS"])

        # Ações
        if st.button("💾 Salvar Sistema"):
            save_system()
        if st.button("📂 Carregar Sistema"):
            load_system()

    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Entradas", "📤 Saídas", "📜 Regras", "👁️ Visualizar"])

    with tab1:
        # Componente de criação de variáveis
        from components.variable_creator import create_input_variables
        create_input_variables()

    with tab2:
        from components.variable_creator import create_output_variables
        create_output_variables()

    with tab3:
        # Construtor de regras
        from components.rule_builder import build_rules
        build_rules()

    with tab4:
        # Visualização completa
        from components.plotter import plot_system
        plot_system()
```

---

### 3. simulator.py - Simulador e Testador

**Responsabilidades:**
- Testar sistema criado com valores específicos
- Visualizar processo de fuzzificação
- Mostrar ativação de regras
- Visualizar defuzzificação
- Gráficos de superfície 3D (para 2 entradas)

**Interface:**

```python
def run():
    st.title("🧪 Simulador de Sistemas Fuzzy")

    # Verificar se há sistema carregado
    if 'fuzzy_system' not in st.session_state:
        st.warning("Nenhum sistema carregado. Vá para o Designer primeiro.")
        if st.button("Ir para Designer"):
            navigate_to('designer')
        return

    system = st.session_state['fuzzy_system']

    # Input de valores
    st.subheader("📊 Valores de Entrada")
    inputs = {}
    for var_name in system.inputs:
        inputs[var_name] = st.slider(
            f"{var_name}:",
            min_value=system.inputs[var_name].universe[0],
            max_value=system.inputs[var_name].universe[1],
            step=0.1
        )

    # Executar simulação
    if st.button("▶️ Executar", type="primary"):
        result = system.evaluate(inputs)

        # Mostrar resultados
        st.success("✅ Simulação concluída!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resultado", f"{result['output']:.2f}")

        with col2:
            # Visualizar fuzzificação
            plot_fuzzification(system, inputs)

        # Visualizar ativação de regras
        plot_rule_activation(system, inputs)

        # Visualizar defuzzificação
        plot_defuzzification(system, result)

    # Superfície 3D (se 2 entradas)
    if len(system.inputs) == 2:
        st.subheader("📈 Superfície de Controle 3D")
        plot_3d_surface(system)
```

---

### 4. exporter.py - Exportação e Importação

**Responsabilidades:**
- Exportar sistema para JSON
- Exportar código Python equivalente
- Importar sistemas salvos
- Compartilhar sistemas

**Interface:**

```python
def run():
    st.title("💾 Exportar/Importar Sistemas")

    tab1, tab2 = st.tabs(["📤 Exportar", "📥 Importar"])

    with tab1:
        st.subheader("Exportar Sistema Atual")

        # Opções de exportação
        export_format = st.radio("Formato:", ["JSON", "Código Python", "Ambos"])

        if st.button("Gerar Exportação"):
            if export_format in ["JSON", "Ambos"]:
                json_data = export_to_json(st.session_state['fuzzy_system'])
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    file_name="fuzzy_system.json",
                    mime="application/json"
                )

            if export_format in ["Código Python", "Ambos"]:
                from components.code_generator import generate_code
                python_code = generate_code(st.session_state['fuzzy_system'])

                st.code(python_code, language='python')
                st.download_button(
                    "📥 Download .py",
                    python_code,
                    file_name="fuzzy_system.py",
                    mime="text/plain"
                )

    with tab2:
        st.subheader("Importar Sistema")

        uploaded_file = st.file_uploader("Escolha um arquivo JSON", type=['json'])

        if uploaded_file:
            import json
            data = json.load(uploaded_file)

            if st.button("Carregar Sistema"):
                load_from_json(data)
                st.success("✅ Sistema carregado com sucesso!")
                navigate_to('designer')
```

---

## 🧩 Componentes Reutilizáveis

### components/variable_creator.py

```python
def create_linguistic_variable():
    """Componente para criar uma variável linguística"""

    with st.expander("➕ Adicionar Nova Variável", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            var_name = st.text_input("Nome da Variável:")
            min_val = st.number_input("Valor Mínimo:", value=0.0)

        with col2:
            var_type = st.selectbox("Tipo:", ["Entrada", "Saída"])
            max_val = st.number_input("Valor Máximo:", value=100.0)

        # Adicionar termos
        st.subheader("Termos Linguísticos")

        num_terms = st.number_input("Número de termos:", min_value=1, max_value=7, value=3)

        for i in range(num_terms):
            with st.container():
                st.markdown(f"**Termo {i+1}**")

                col1, col2 = st.columns(2)
                with col1:
                    term_name = st.text_input(f"Nome:", key=f"term_name_{i}")
                    mf_type = st.selectbox("Função:",
                                          ["triangular", "trapezoidal", "gaussian", "sigmoid"],
                                          key=f"mf_type_{i}")

                with col2:
                    # Parâmetros dependem do tipo de função
                    if mf_type == "triangular":
                        a = st.number_input("a:", key=f"param_a_{i}")
                        b = st.number_input("b:", key=f"param_b_{i}")
                        c = st.number_input("c:", key=f"param_c_{i}")
                        params = (a, b, c)

                    # ... outros tipos

                # Visualização em tempo real
                plot_membership_function(mf_type, params, (min_val, max_val))

        if st.button("✅ Criar Variável"):
            # Criar usando pyfuzzy-toolbox
            from fuzzy_systems.core import LinguisticVariable

            var = LinguisticVariable(name=var_name, universe=(min_val, max_val))
            # Adicionar termos...

            # Salvar no session_state
            if 'variables' not in st.session_state:
                st.session_state['variables'] = {}
            st.session_state['variables'][var_name] = var

            st.success(f"✅ Variável '{var_name}' criada!")
```

### components/rule_builder.py

```python
def build_rules():
    """Interface visual para construir regras fuzzy"""

    st.subheader("📜 Construtor de Regras")

    # Verificar se há variáveis
    if 'variables' not in st.session_state or not st.session_state['variables']:
        st.warning("Adicione variáveis de entrada e saída primeiro!")
        return

    # Separar inputs e outputs
    inputs = {k: v for k, v in st.session_state['variables'].items()
              if v.is_input}
    outputs = {k: v for k, v in st.session_state['variables'].items()
               if not v.is_input}

    # Interface de criação de regra
    with st.expander("➕ Adicionar Nova Regra", expanded=True):
        st.markdown("**SE**")

        # Antecedentes
        antecedents = []
        for i, (var_name, var) in enumerate(inputs.items()):
            col1, col2 = st.columns([1, 3])

            with col1:
                if i > 0:
                    st.markdown("**E**")

            with col2:
                term = st.selectbox(
                    f"{var_name} é:",
                    list(var.terms.keys()),
                    key=f"antecedent_{var_name}"
                )
                antecedents.append(term)

        st.markdown("**ENTÃO**")

        # Consequentes
        consequents = []
        for var_name, var in outputs.items():
            term = st.selectbox(
                f"{var_name} é:",
                list(var.terms.keys()),
                key=f"consequent_{var_name}"
            )
            consequents.append(term)

        if st.button("➕ Adicionar Regra"):
            # Criar regra
            rule = (antecedents, consequents)

            if 'rules' not in st.session_state:
                st.session_state['rules'] = []
            st.session_state['rules'].append(rule)

            st.success("✅ Regra adicionada!")

    # Mostrar regras existentes
    if 'rules' in st.session_state and st.session_state['rules']:
        st.subheader("Regras Definidas")

        for i, rule in enumerate(st.session_state['rules']):
            col1, col2 = st.columns([5, 1])

            with col1:
                rule_text = format_rule(rule, inputs, outputs)
                st.text(f"{i+1}. {rule_text}")

            with col2:
                if st.button("🗑️", key=f"delete_rule_{i}"):
                    st.session_state['rules'].pop(i)
                    st.rerun()
```

### components/code_generator.py

```python
def generate_code(system):
    """Gera código Python equivalente ao sistema visual"""

    code = f"""\"\"\"
Sistema Fuzzy gerado automaticamente pela interface pyfuzzy-toolbox
Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}
\"\"\"

import fuzzy_systems as fs

# Criar sistema
system = fs.{system.type}System()

# Adicionar variáveis de entrada
"""

    # Gerar código para inputs
    for var_name, var in system.inputs.items():
        code += f"\nsystem.add_input('{var_name}', {var.universe})"

        for term_name, term in var.terms.items():
            code += f"\nsystem.add_term('{var_name}', '{term_name}', '{term.mf_type}', {term.params})"

    # Gerar código para outputs
    code += "\n\n# Adicionar variável(is) de saída\n"
    for var_name, var in system.outputs.items():
        code += f"system.add_output('{var_name}', {var.universe})\n"

        for term_name, term in var.terms.items():
            code += f"system.add_term('{var_name}', '{term_name}', '{term.mf_type}', {term.params})\n"

    # Gerar código para regras
    code += "\n# Adicionar regras\n"
    code += "system.add_rules([\n"
    for rule in system.rules:
        code += f"    {rule},\n"
    code += "])\n"

    # Exemplo de uso
    code += """
# Exemplo de uso
if __name__ == '__main__':
    # Avaliar sistema
    result = system.evaluate({
        # Adicione seus valores de entrada aqui
    })

    print(f"Resultado: {result}")
"""

    return code
```

---

## 📊 Estado da Aplicação (session_state)

```python
st.session_state = {
    'page': 'home',                    # Página atual
    'fuzzy_system': None,              # Sistema fuzzy atual (objeto MamdaniSystem/SugenoSystem)
    'system_type': 'Mamdani',          # Tipo de sistema
    'variables': {},                    # Dicionário de variáveis linguísticas
    'rules': [],                        # Lista de regras
    'current_simulation': None,         # Resultados da última simulação
    'history': [],                      # Histórico de simulações
}
```

---

## 🎯 Features Avançadas (Fase 2)

1. **Otimização Interativa**
   - Interface para ANFIS training
   - Visualização da curva de aprendizado
   - Ajuste de hiperparâmetros

2. **Comparação de Sistemas**
   - Carregar múltiplos sistemas
   - Comparar performance
   - Visualizações lado a lado

3. **Datasets**
   - Importar CSV para treinamento
   - Validação cruzada visual
   - Métricas de performance (RMSE, MAE, etc.)

4. **Temas e Customização**
   - Tema claro/escuro
   - Cores customizáveis para plots
   - Salvar preferências

5. **Colaboração**
   - Compartilhar sistemas via link
   - Galeria de sistemas públicos
   - Comentários e votação

---

## 🚀 Roadmap de Implementação

### Sprint 1: Fundação (1-2 semanas)
- [ ] Setup inicial do projeto
- [ ] Estrutura de pastas
- [ ] Navegação SPA básica
- [ ] Módulo home.py
- [ ] Session state manager

### Sprint 2: Designer Básico (2-3 semanas)
- [ ] Interface de criação de variáveis
- [ ] Editor de termos linguísticos
- [ ] Visualização de funções de pertinência
- [ ] Salvar/carregar sistemas (JSON)

### Sprint 3: Regras e Simulação (2 semanas)
- [ ] Construtor de regras visual
- [ ] Módulo simulator.py
- [ ] Visualização de fuzzificação
- [ ] Visualização de defuzzificação

### Sprint 4: Export e Polimento (1-2 semanas)
- [ ] Gerador de código Python
- [ ] Módulo exporter.py
- [ ] Exemplos pré-carregados
- [ ] Documentação e tutoriais

### Sprint 5: Features Avançadas (2-3 semanas)
- [ ] Suporte a ANFIS
- [ ] Importação de datasets
- [ ] Superfícies 3D
- [ ] Otimização de parâmetros

---

## 📦 Dependências (requirements.txt)

```txt
pyfuzzy-toolbox>=1.0.0
streamlit>=1.28.0
numpy>=1.20.0
matplotlib>=3.3.0
plotly>=5.0.0           # Para gráficos interativos 3D
pandas>=1.2.0           # Para importação de dados
```

---

## 🧪 Exemplo de Uso Final

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run main.py
```

**Workflow do usuário:**

1. Abrir aplicação → Página Home
2. Clicar em "Designer" → Criar sistema Mamdani
3. Adicionar variável "temperatura" (0-40°C) com termos: fria, morna, quente
4. Adicionar variável "velocidade_ventilador" (0-100%) com termos: lento, médio, rápido
5. Criar regras:
   - SE temperatura é fria ENTÃO velocidade é lento
   - SE temperatura é morna ENTÃO velocidade é médio
   - SE temperatura é quente ENTÃO velocidade é rápido
6. Ir para "Simulador" → Testar com temperatura = 28°C
7. Ver resultado: velocidade = 65%
8. Ir para "Exportar" → Download do código Python
9. Usar código em projeto próprio!

---

## 📝 Notas Finais

- **Prioridade 1**: Interface simples e intuitiva para iniciantes
- **Prioridade 2**: Gerar código Python limpo e documentado
- **Prioridade 3**: Performance e responsividade
- **Prioridade 4**: Features avançadas (ANFIS, otimização)

Este plano serve como guia de implementação e pode ser ajustado conforme necessidades específicas surgem durante o desenvolvimento.

---

**Autor:** Claude Code
**Biblioteca:** pyfuzzy-toolbox v1.0.0
**GitHub:** https://github.com/1moi6/pyfuzzy-toolbox
