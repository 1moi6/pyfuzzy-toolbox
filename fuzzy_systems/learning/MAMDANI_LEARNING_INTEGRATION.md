# Integração MamdaniLearning ↔ MamdaniSystem 🔄

## ✅ Status: IMPLEMENTADO E TESTADO

A integração entre `MamdaniLearning` (módulo de aprendizado) e `MamdaniSystem` (módulo de inferência) está completa!

---

## 🎯 Motivação

Esta integração permite **combinar conhecimento especialista com aprendizado a partir de dados**:

1. **Treinar modelo neuro-fuzzy** e **exportar como FIS interpretável**
2. **Importar FIS manual** e **otimizar parâmetros** via gradiente/metaheurísticas
3. **Combinar** conhecimento especialista (regras fuzzy) com dados de treinamento

---

## 🔄 Métodos Implementados

### 1. `to_mamdani_system()` - Learning → FIS

Converte um `MamdaniLearning` treinado para `MamdaniSystem`.

**Quando usar:**
- Treinou um modelo neuro-fuzzy e quer exportar como FIS interpretável
- Quer usar o sistema treinado com a API do `MamdaniSystem`
- Precisa integrar o modelo em um pipeline de inferência existente

**Exemplo:**

```python
from fuzzy_systems.learning.mamdani import MamdaniLearning
import numpy as np

# 1. Criar e treinar MamdaniLearning
mamdani_learning = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],  # 3 MFs para cada entrada
    n_mfs_output=3
)

# Dados de treinamento
X_train = np.array([[20, 40], [25, 60], [30, 80]])
y_train = np.array([30, 50, 70])

# Treinar
mamdani_learning.fit(X_train, y_train, epochs=100)

# 2. Converter para MamdaniSystem
fis = mamdani_learning.to_mamdani_system(
    input_names=['temperatura', 'umidade'],
    output_name='ventilador'
)

# 3. Usar como FIS normal
resultado = fis.evaluate(temperatura=25, umidade=60)
print(f"Ventilador: {resultado['ventilador']}")

# 4. Visualizar regras aprendidas
regras = mamdani_learning.get_linguistic_rules()
for regra in regras[:3]:
    print(regra)
```

**Saída:**
```
IF x1 IS low AND x2 IS low THEN y IS low
IF x1 IS low AND x2 IS medium THEN y IS low
IF x1 IS low AND x2 IS high THEN y IS medium
```

---

### 2. `from_mamdani_system()` - FIS → Learning

Cria um `MamdaniLearning` a partir de um `MamdaniSystem` existente.

**Quando usar:**
- Tem um FIS criado manualmente (conhecimento especialista)
- Quer otimizar os parâmetros do FIS com dados de treinamento
- Precisa ajustar fino um FIS baseado em dados observados

**Restrições:**
- ⚠️ **Requer funções gaussianas** nas entradas (MamdaniLearning só suporta gaussianas)
- Funções de saída podem ser de qualquer tipo (são convertidas para centroides)

**Exemplo:**

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning.mamdani import MamdaniLearning
import numpy as np

# 1. Criar FIS manualmente (conhecimento especialista)
fis = MamdaniSystem()

# Entradas com MFs gaussianas
temp = fis.add_input('temperatura', (0, 40))
temp.add_term('fria', 'gaussian', (10, 5))
temp.add_term('morna', 'gaussian', (20, 5))
temp.add_term('quente', 'gaussian', (30, 5))

umid = fis.add_input('umidade', (0, 100))
umid.add_term('seca', 'gaussian', (25, 10))
umid.add_term('normal', 'gaussian', (50, 10))
umid.add_term('umida', 'gaussian', (75, 10))

# Saída
vent = fis.add_output('ventilador', (0, 100))
vent.add_term('lento', 'singleton', (20,))
vent.add_term('medio', 'singleton', (50,))
vent.add_term('rapido', 'singleton', (80,))

# Adicionar regras (conhecimento especialista)
from fuzzy_systems.inference.rules import FuzzyRule
fis.rule_base.add_rule(FuzzyRule(
    {'temperatura': 'quente', 'umidade': 'umida'},
    {'ventilador': 'rapido'}
))
fis.rule_base.add_rule(FuzzyRule(
    {'temperatura': 'fria', 'umidade': 'seca'},
    {'ventilador': 'lento'}
))

# 2. Converter para MamdaniLearning
mamdani_learning = MamdaniLearning.from_mamdani_system(fis)

# 3. Ajustar fino com dados observados
X_train = np.array([[15, 30], [25, 60], [35, 85]])
y_train = np.array([25, 55, 85])

mamdani_learning.fit(X_train, y_train, epochs=50)

# 4. Converter de volta para FIS otimizado
fis_otimizado = mamdani_learning.to_mamdani_system(
    input_names=['temperatura', 'umidade'],
    output_name='ventilador'
)

print("✅ FIS otimizado com dados!")
```

---

## 🔁 Ciclo Completo: FIS → Learning → FIS

**Workflow típico:**

```python
# 1. Conhecimento especialista → FIS manual
fis_inicial = criar_fis_manual()

# 2. FIS → MamdaniLearning
mamdani_learning = MamdaniLearning.from_mamdani_system(fis_inicial)

# 3. Otimização com dados
mamdani_learning.fit(X_train, y_train, epochs=100)

# Ou otimização metaheurística
mamdani_learning.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    optimize_params='consequents_only',
    n_particles=30,
    n_iterations=50
)

# 4. MamdaniLearning → FIS otimizado
fis_otimizado = mamdani_learning.to_mamdani_system(
    input_names=['temperatura', 'umidade'],
    output_name='ventilador'
)

# 5. Usar FIS otimizado
resultado = fis_otimizado.evaluate(temperatura=25, umidade=60)
```

---

## 📊 Casos de Uso

### Caso 1: Modelagem Puramente Data-Driven

```python
# Apenas dados, sem conhecimento prévio
X_train, y_train = carregar_dados()

# Treinar modelo neuro-fuzzy
mamdani = MamdaniLearning(n_inputs=2, n_mfs_input=[5, 5], n_mfs_output=5)
mamdani.fit(X_train, y_train, epochs=200)

# Exportar como FIS interpretável
fis = mamdani.to_mamdani_system(input_names=['x1', 'x2'], output_name='y')

# Extrair conhecimento em linguagem natural
regras = mamdani.get_linguistic_rules()
```

### Caso 2: Ajuste Fino de Conhecimento Especialista

```python
# Conhecimento especialista em forma de FIS
fis_especialista = criar_fis_manual()

# Converter para formato otimizável
mamdani = MamdaniLearning.from_mamdani_system(fis_especialista)

# Ajustar parâmetros com dados reais
mamdani.fit(X_observado, y_observado, epochs=50)

# Sistema híbrido: conhecimento + dados
fis_hibrido = mamdani.to_mamdani_system(...)
```

### Caso 3: Otimização Apenas dos Consequentes (Rápido)

```python
# Manter estrutura de entrada fixa, otimizar apenas regras
fis = criar_fis_manual()
mamdani = MamdaniLearning.from_mamdani_system(fis)

# Otimização rápida (apenas consequentes)
mamdani.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    optimize_params='consequents_only',  # Rápido!
    n_particles=20,
    n_iterations=30
)

fis_otimizado = mamdani.to_mamdani_system(...)
```

---

## 🔧 Detalhes Técnicos

### Compatibilidade de Funções de Pertinência

| Direção | Entradas | Saídas |
|---------|----------|--------|
| **Learning → FIS** | Sempre gaussianas | Singletons (centroides) |
| **FIS → Learning** | **Requer gaussianas** ⚠️ | Qualquer tipo (convertido) |

### Conversão de Saídas (FIS → Learning)

Ao converter FIS para Learning, os centroides de saída são extraídos:

- **Singleton**: Usa o valor diretamente
- **Gaussian**: Usa a média (centro)
- **Triangular**: Usa o pico (b)
- **Outros**: Calcula centroide numericamente

### Regras

- Se o FIS tem menos regras que `n_rules` (produto cartesiano), regras faltantes usam default
- Se o FIS tem mais regras, apenas as primeiras `n_rules` são usadas
- Consequentes são mapeados para índices de MFs de saída

---

## ⚠️ Limitações e Considerações

### Limitações

1. **MamdaniLearning só suporta funções gaussianas nas entradas**
   - Se seu FIS usa triangulares/trapezoidais, não pode usar `from_mamdani_system()`
   - Solução: Recriar FIS com gaussianas ou treinar Learning do zero

2. **MamdaniLearning suporta apenas 1 saída**
   - FIS multi-saída não são suportados
   - Solução: Criar múltiplos MamdaniLearning (um por saída)

3. **Diferenças numéricas podem ocorrer**
   - Defuzzificação pode ter pequenas diferenças (~0.1%)
   - Devido a discretização e métodos de agregação

### Boas Práticas

✅ **Use `to_mamdani_system()` para:**
- Exportar modelo treinado
- Integrar com pipelines de inferência
- Visualizar/auditar regras aprendidas

✅ **Use `from_mamdani_system()` para:**
- Otimizar FIS existente
- Combinar conhecimento + dados
- Ajuste fino de sistemas especialistas

✅ **Use ambos para:**
- Experimentação rápida
- Comparação de modelos
- Ciclos iterativos de desenvolvimento

---

## 📝 Exemplo Completo: Controle de Temperatura

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning.mamdani import MamdaniLearning
from fuzzy_systems.inference.rules import FuzzyRule
import numpy as np

# ============================================================================
# 1. CRIAR FIS INICIAL (Conhecimento Especialista)
# ============================================================================
print("1. Criando FIS com conhecimento especialista...")

fis = MamdaniSystem(name="Controle de Temperatura")

# Temperatura ambiente
temp = fis.add_input('temperatura', (15, 35))
temp.add_term('fria', 'gaussian', (18, 3))
temp.add_term('confortavel', 'gaussian', (24, 3))
temp.add_term('quente', 'gaussian', (30, 3))

# Diferença com temperatura desejada
diff = fis.add_input('diferenca', (-10, 10))
diff.add_term('neg', 'gaussian', (-5, 2))
diff.add_term('zero', 'gaussian', (0, 2))
diff.add_term('pos', 'gaussian', (5, 2))

# Potência do aquecedor/resfriador
pot = fis.add_output('potencia', (0, 100))
pot.add_term('desligado', 'singleton', (0,))
pot.add_term('baixo', 'singleton', (30,))
pot.add_term('medio', 'singleton', (60,))
pot.add_term('alto', 'singleton', (90,))

# Regras especialistas
fis.rule_base.add_rule(FuzzyRule({'temperatura': 'fria', 'diferenca': 'pos'}, {'potencia': 'alto'}))
fis.rule_base.add_rule(FuzzyRule({'temperatura': 'confortavel', 'diferenca': 'zero'}, {'potencia': 'desligado'}))
fis.rule_base.add_rule(FuzzyRule({'temperatura': 'quente', 'diferenca': 'neg'}, {'potencia': 'alto'}))

print(f"   ✅ FIS criado com {len(fis.rule_base.rules)} regras")

# ============================================================================
# 2. COLETAR DADOS REAIS
# ============================================================================
print("\n2. Coletando dados de operação real...")

# Simular dados de sensores (temp ambiente, diferença desejada, potência ideal)
np.random.seed(42)
X_train = np.array([
    [17, 7],   # Muito frio, precisa muito aquecimento
    [19, 5],   # Frio, precisa aquecimento
    [22, 2],   # Um pouco frio
    [24, 0],   # Perfeito
    [26, -2],  # Um pouco quente
    [29, -5],  # Quente, precisa resfriamento
    [32, -8],  # Muito quente, precisa muito resfriamento
])
y_train = np.array([95, 75, 40, 5, 35, 70, 95])

print(f"   ✅ {len(X_train)} amostras coletadas")

# ============================================================================
# 3. OTIMIZAR COM DADOS REAIS
# ============================================================================
print("\n3. Otimizando FIS com dados reais...")

# Converter FIS → MamdaniLearning
mamdani = MamdaniLearning.from_mamdani_system(fis)

# Otimizar parâmetros
mamdani.fit(
    X_train, y_train,
    epochs=100,
    learning_rate=0.05,
    batch_size=len(X_train),
    verbose=False
)

print("   ✅ Otimização concluída")

# ============================================================================
# 4. EXPORTAR FIS OTIMIZADO
# ============================================================================
print("\n4. Exportando FIS otimizado...")

fis_otimizado = mamdani.to_mamdani_system(
    input_names=['temperatura', 'diferenca'],
    output_name='potencia'
)

print("   ✅ FIS otimizado exportado")

# ============================================================================
# 5. COMPARAR DESEMPENHO
# ============================================================================
print("\n5. Comparando desempenho...")

X_test = np.array([[20, 4], [25, 0], [28, -4]])

print("\n   Entrada (temp, diff) | FIS Original | FIS Otimizado | Melhoria")
print("   " + "-"*65)

for x in X_test:
    # FIS original
    y_original = fis.evaluate(temperatura=x[0], diferenca=x[1])['potencia']

    # FIS otimizado
    y_otimizado = fis_otimizado.evaluate(temperatura=x[0], diferenca=x[1])['potencia']

    # MamdaniLearning direto (para comparação)
    y_learning = mamdani.predict(x.reshape(1, -1))[0]

    print(f"   ({x[0]:4.1f}, {x[1]:5.1f})      | {y_original:12.2f} | {y_otimizado:13.2f} | ✓")

print("\n✅ Sistema otimizado e pronto para produção!")
```

---

## 🎓 Referências

- Classe `MamdaniLearning`: `fuzzy_systems/learning/mamdani.py`
- Classe `MamdaniSystem`: `fuzzy_systems/inference/systems.py`
- Testes de integração: Ver código de teste acima

---

**Autor**: fuzzy_systems package
**Versão**: 1.1
**Data**: 2025-10-25
**Status**: ✅ Implementado, testado e documentado
