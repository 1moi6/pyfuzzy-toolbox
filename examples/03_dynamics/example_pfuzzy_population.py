"""
Exemplo: Modelo de Crescimento Populacional com p-Fuzzy
=========================================================

Demonstra o uso de sistemas p-fuzzy para modelar dinâmica populacional
onde a taxa de crescimento depende da densidade populacional atual.

Modelo clássico: Equação Logística
dx/dt = r*x*(1 - x/K)

onde:
- x: população
- r: taxa de crescimento intrínseca
- K: capacidade de carga

Aqui implementamos uma versão fuzzy onde as regras determinam
a taxa de crescimento baseada na população atual.
"""

import sys
sys.path.insert(0, '/Users/1moi6/Desktop/Minicurso Fuzzy/fuzzy_systems')

import numpy as np
import fuzzy_systems as fs
import matplotlib.pyplot as plt

print("=" * 70)
print("EXEMPLO: Crescimento Populacional com p-Fuzzy")
print("=" * 70)

# ============================================================================
# PARTE 1: Criar Sistema de Inferência Fuzzy
# ============================================================================

print("\n1️⃣  Criando Sistema de Inferência Fuzzy...")

# Sistema Mamdani para determinar taxa de crescimento
fis = fs.MamdaniSystem(name="População Logística Fuzzy")

# Entrada: população atual (normalizada, 0 a 100)
fis.add_input('population', (0, 100))
fis.add_term('population', 'baixa', 'gaussian', (0, 15))
fis.add_term('population', 'media', 'gaussian', (50, 15))
fis.add_term('population', 'alta', 'gaussian', (100, 15))

# Saída: taxa de crescimento (pode ser negativa)
fis.add_output('growth_rate', (-5, 10))
fis.add_term('growth_rate', 'declinio', 'triangular', (-5, -5, 0))
fis.add_term('growth_rate', 'estavel', 'triangular', (-2, 0, 2))
fis.add_term('growth_rate', 'crescimento', 'triangular', (0, 10, 10))

# Regras: População baixa -> crescimento alto
#         População média -> crescimento moderado
#         População alta -> declínio
fis.add_rule({
    'population': 'baixa'
}, {
    'growth_rate': 'crescimento'
})

fis.add_rule({
    'population': 'media'
}, {
    'growth_rate': 'estavel'
})

fis.add_rule({
    'population': 'alta'
}, {
    'growth_rate': 'declinio'
})

print(f"✅ FIS criado:")
print(f"   • {len(fis.input_variables)} entrada: {list(fis.input_variables.keys())}")
print(f"   • {len(fis.output_variables)} saída: {list(fis.output_variables.keys())}")
print(f"   • {len(fis.rule_base.rules)} regras")


# ============================================================================
# PARTE 2: Sistema Discreto (modo absoluto)
# ============================================================================

print("\n\n" + "=" * 70)
print("2️⃣  Sistema DISCRETO (modo absoluto)")
print("=" * 70)
print("Equação: x_{n+1} = x_n + f(x_n)")

# Criar sistema p-fuzzy discreto
pfuzzy_discrete = fs.dynamic.PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['population'],
    dt=0.1  # Passo de tempo
)

# Simular
print("\n📊 Simulando 100 iterações com x0 = 10...")
trajectory_discrete = pfuzzy_discrete.simulate(
    x0={'population': 10},
    n_steps=100
)

print(f"✅ Simulação concluída!")
print(f"   • População inicial: {trajectory_discrete[0, 0]:.2f}")
print(f"   • População final: {trajectory_discrete[-1, 0]:.2f}")
print(f"   • População máxima: {trajectory_discrete[:, 0].max():.2f}")


# ============================================================================
# PARTE 3: Sistema Contínuo (modo absoluto)
# ============================================================================

print("\n\n" + "=" * 70)
print("3️⃣  Sistema CONTÍNUO (modo absoluto)")
print("=" * 70)
print("Equação: dx/dt = f(x)")

# Criar sistema p-fuzzy contínuo
pfuzzy_continuous = fs.dynamic.PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['population'],
    method='rk4'  # Runge-Kutta 4ª ordem
)

# Simular
print("\n📊 Simulando de t=0 a t=10 com x0 = 10...")
trajectory_continuous = pfuzzy_continuous.simulate(
    x0={'population': 10},
    t_span=(0, 10),
    dt=0.01
)

print(f"✅ Simulação concluída!")
print(f"   • População inicial: {trajectory_continuous[0, 0]:.2f}")
print(f"   • População final: {trajectory_continuous[-1, 0]:.2f}")
print(f"   • População máxima: {trajectory_continuous[:, 0].max():.2f}")


# ============================================================================
# PARTE 4: Sistema Contínuo (modo relativo)
# ============================================================================

print("\n\n" + "=" * 70)
print("4️⃣  Sistema CONTÍNUO (modo relativo)")
print("=" * 70)
print("Equação: dx/dt = x * f(x)")

# Criar sistema com modo relativo
pfuzzy_relative = fs.dynamic.PFuzzyContinuous(
    fis=fis,
    mode='relative',
    state_vars=['population'],
    method='rk4'
)

# Simular
print("\n📊 Simulando de t=0 a t=10 com x0 = 10...")
trajectory_relative = pfuzzy_relative.simulate(
    x0={'population': 10},
    t_span=(0, 10),
    dt=0.01
)

print(f"✅ Simulação concluída!")
print(f"   • População inicial: {trajectory_relative[0, 0]:.2f}")
print(f"   • População final: {trajectory_relative[-1, 0]:.2f}")
print(f"   • População máxima: {trajectory_relative[:, 0].max():.2f}")


# ============================================================================
# PARTE 5: Visualização Comparativa
# ============================================================================

print("\n\n" + "=" * 70)
print("5️⃣  Visualização Comparativa")
print("=" * 70)

# Criar figura com 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Comparação Discreto vs Contínuo (modo absoluto)
ax1.plot(pfuzzy_discrete.time, pfuzzy_discrete.trajectory,
         'b-o', markersize=4, label='Discreto (absoluto)', linewidth=2)
ax1.plot(pfuzzy_continuous.time, pfuzzy_continuous.trajectory,
         'r-', label='Contínuo (absoluto)', linewidth=2)
ax1.set_xlabel('Tempo', fontsize=12)
ax1.set_ylabel('População', fontsize=12)
ax1.set_title('Discreto vs Contínuo (modo absoluto)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Comparação Absoluto vs Relativo (contínuo)
ax2.plot(pfuzzy_continuous.time, pfuzzy_continuous.trajectory,
         'g-', label='Absoluto: dx/dt = f(x)', linewidth=2)
ax2.plot(pfuzzy_relative.time, pfuzzy_relative.trajectory,
         'm-', label='Relativo: dx/dt = x*f(x)', linewidth=2)
ax2.set_xlabel('Tempo', fontsize=12)
ax2.set_ylabel('População', fontsize=12)
ax2.set_title('Modo Absoluto vs Relativo', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/pfuzzy_population_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráfico salvo em: /tmp/pfuzzy_population_comparison.png")


# ============================================================================
# PARTE 6: Análise da Superfície de Controle
# ============================================================================

print("\n\n" + "=" * 70)
print("6️⃣  Superfície de Controle Fuzzy")
print("=" * 70)

# Avaliar FIS para diferentes valores de população
pop_values = np.linspace(0, 100, 100)
growth_values = []

for pop in pop_values:
    result = fis.evaluate({'population': pop})
    growth_values.append(result['growth_rate'])

# Plot da superfície
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pop_values, growth_values, 'b-', linewidth=3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('População', fontsize=12)
ax.set_ylabel('Taxa de Crescimento', fontsize=12)
ax.set_title('Superfície de Controle: f(população) → taxa de crescimento',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/pfuzzy_control_surface.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráfico salvo em: /tmp/pfuzzy_control_surface.png")


# ============================================================================
# RESUMO
# ============================================================================

print("\n\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

print("""
✅ SISTEMAS P-FUZZY IMPLEMENTADOS:

1. Sistema Discreto (absoluto):
   x_{n+1} = x_n + f(x_n)
   → Mudança absoluta a cada iteração

2. Sistema Contínuo (absoluto):
   dx/dt = f(x)
   → Taxa de mudança determinada pelo FIS

3. Sistema Contínuo (relativo):
   dx/dt = x * f(x)
   → Taxa proporcional ao estado atual

🔑 CONCEITOS-CHAVE:

• O FIS mapeia estado → taxa de mudança
• Regras fuzzy codificam dinâmica do sistema
• Diferentes modos capturam diferentes comportamentos
• Integração numérica (Euler/RK4) para sistemas contínuos

📊 APLICAÇÕES:

• Modelos populacionais
• Dinâmica de epidemias
• Sistemas de controle
• Processos biológicos
• Dinâmica econômica

🎯 PRÓXIMOS PASSOS:

1. Experimente com diferentes regras fuzzy
2. Teste outros métodos de integração
3. Compare com modelos clássicos (logístico, Lotka-Volterra)
4. Adicione múltiplas variáveis de estado
""")

print("=" * 70)
print("✅ Exemplo concluído!")
print("=" * 70)
