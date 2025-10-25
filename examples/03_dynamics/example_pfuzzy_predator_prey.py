"""
Exemplo: Sistema Predador-Presa (Lotka-Volterra) com p-Fuzzy
=============================================================

Implementa uma versão fuzzy do clássico modelo Lotka-Volterra
de dinâmica predador-presa usando sistemas p-fuzzy.

Modelo clássico:
dx/dt = αx - βxy    (presas)
dy/dt = δxy - γy    (predadores)

Versão fuzzy:
As taxas de mudança são determinadas por regras fuzzy baseadas
nas populações atuais de presas e predadores.
"""

import sys
sys.path.insert(0, '/Users/1moi6/Desktop/Minicurso Fuzzy/fuzzy_systems')

import numpy as np
import fuzzy_systems as fs
import matplotlib.pyplot as plt

print("=" * 70)
print("EXEMPLO: Sistema Predador-Presa Fuzzy (Lotka-Volterra)")
print("=" * 70)

# ============================================================================
# PARTE 1: Criar FIS para Presas
# ============================================================================

print("\n1️⃣  Criando FIS para dinâmica das PRESAS...")

fis_prey = fs.MamdaniSystem(name="Dinâmica Presas")

# Entradas: população de presas e predadores
fis_prey.add_input('prey', (0, 100))
fis_prey.add_term('prey', 'baixa', 'gaussian', (10, 15))
fis_prey.add_term('prey', 'media', 'gaussian', (50, 15))
fis_prey.add_term('prey', 'alta', 'gaussian', (90, 15))

fis_prey.add_input('predator', (0, 100))
fis_prey.add_term('predator', 'baixa', 'gaussian', (10, 15))
fis_prey.add_term('predator', 'media', 'gaussian', (50, 15))
fis_prey.add_term('predator', 'alta', 'gaussian', (90, 15))

# Saída: taxa de mudança das presas
fis_prey.add_output('prey', (-20, 20))
fis_prey.add_term('prey', 'declinio_forte', 'triangular', (-20, -20, -5))
fis_prey.add_term('prey', 'declinio', 'triangular', (-10, -5, 0))
fis_prey.add_term('prey', 'estavel', 'triangular', (-3, 0, 3))
fis_prey.add_term('prey', 'crescimento', 'triangular', (0, 5, 10))
fis_prey.add_term('prey', 'crescimento_forte', 'triangular', (5, 20, 20))

# Regras para presas:
# - Muitos predadores → declínio das presas
# - Poucos predadores → crescimento das presas
# - Muitas presas → crescimento (recursos abundantes)

fis_prey.add_rule({'prey': 'baixa', 'predator': 'baixa'},
                  {'prey': 'crescimento_forte'})
fis_prey.add_rule({'prey': 'baixa', 'predator': 'media'},
                  {'prey': 'declinio'})
fis_prey.add_rule({'prey': 'baixa', 'predator': 'alta'},
                  {'prey': 'declinio_forte'})

fis_prey.add_rule({'prey': 'media', 'predator': 'baixa'},
                  {'prey': 'crescimento'})
fis_prey.add_rule({'prey': 'media', 'predator': 'media'},
                  {'prey': 'estavel'})
fis_prey.add_rule({'prey': 'media', 'predator': 'alta'},
                  {'prey': 'declinio_forte'})

fis_prey.add_rule({'prey': 'alta', 'predator': 'baixa'},
                  {'prey': 'crescimento'})
fis_prey.add_rule({'prey': 'alta', 'predator': 'media'},
                  {'prey': 'estavel'})
fis_prey.add_rule({'prey': 'alta', 'predator': 'alta'},
                  {'prey': 'declinio'})

print(f"✅ FIS Presas: {len(fis_prey.rule_base.rules)} regras")


# ============================================================================
# PARTE 2: Criar FIS para Predadores
# ============================================================================

print("\n2️⃣  Criando FIS para dinâmica dos PREDADORES...")

fis_predator = fs.MamdaniSystem(name="Dinâmica Predadores")

# Mesmas entradas
fis_predator.add_input('prey', (0, 100))
fis_predator.add_term('prey', 'baixa', 'gaussian', (10, 15))
fis_predator.add_term('prey', 'media', 'gaussian', (50, 15))
fis_predator.add_term('prey', 'alta', 'gaussian', (90, 15))

fis_predator.add_input('predator', (0, 100))
fis_predator.add_term('predator', 'baixa', 'gaussian', (10, 15))
fis_predator.add_term('predator', 'media', 'gaussian', (50, 15))
fis_predator.add_term('predator', 'alta', 'gaussian', (90, 15))

# Saída: taxa de mudança dos predadores
fis_predator.add_output('predator', (-20, 20))
fis_predator.add_term('predator', 'declinio_forte', 'triangular', (-20, -20, -5))
fis_predator.add_term('predator', 'declinio', 'triangular', (-10, -5, 0))
fis_predator.add_term('predator', 'estavel', 'triangular', (-3, 0, 3))
fis_predator.add_term('predator', 'crescimento', 'triangular', (0, 5, 10))
fis_predator.add_term('predator', 'crescimento_forte', 'triangular', (5, 20, 20))

# Regras para predadores:
# - Muitas presas → crescimento dos predadores
# - Poucas presas → declínio dos predadores
# - Muitos predadores → competição, crescimento menor

fis_predator.add_rule({'prey': 'baixa', 'predator': 'baixa'},
                      {'predator': 'declinio'})
fis_predator.add_rule({'prey': 'baixa', 'predator': 'media'},
                      {'predator': 'declinio_forte'})
fis_predator.add_rule({'prey': 'baixa', 'predator': 'alta'},
                      {'predator': 'declinio_forte'})

fis_predator.add_rule({'prey': 'media', 'predator': 'baixa'},
                      {'predator': 'crescimento'})
fis_predator.add_rule({'prey': 'media', 'predator': 'media'},
                      {'predator': 'estavel'})
fis_predator.add_rule({'prey': 'media', 'predator': 'alta'},
                      {'predator': 'declinio'})

fis_predator.add_rule({'prey': 'alta', 'predator': 'baixa'},
                      {'predator': 'crescimento_forte'})
fis_predator.add_rule({'prey': 'alta', 'predator': 'media'},
                      {'predator': 'crescimento'})
fis_predator.add_rule({'prey': 'alta', 'predator': 'alta'},
                      {'predator': 'estavel'})

print(f"✅ FIS Predadores: {len(fis_predator.rule_base.rules)} regras")


# ============================================================================
# PARTE 3: Criar Sistema p-Fuzzy Acoplado
# ============================================================================

print("\n\n" + "=" * 70)
print("3️⃣  Criando Sistema p-Fuzzy Acoplado")
print("=" * 70)

# Criar um FIS combinado com duas saídas
fis_combined = fs.MamdaniSystem(name="Lotka-Volterra Fuzzy")

# Entradas
fis_combined.add_input('prey', (0, 100))
fis_combined.add_term('prey', 'baixa', 'gaussian', (10, 15))
fis_combined.add_term('prey', 'media', 'gaussian', (50, 15))
fis_combined.add_term('prey', 'alta', 'gaussian', (90, 15))

fis_combined.add_input('predator', (0, 100))
fis_combined.add_term('predator', 'baixa', 'gaussian', (10, 15))
fis_combined.add_term('predator', 'media', 'gaussian', (50, 15))
fis_combined.add_term('predator', 'alta', 'gaussian', (90, 15))

# Duas saídas
fis_combined.add_output('prey', (-20, 20))
fis_combined.add_term('prey', 'declinio_forte', 'triangular', (-20, -20, -5))
fis_combined.add_term('prey', 'declinio', 'triangular', (-10, -5, 0))
fis_combined.add_term('prey', 'estavel', 'triangular', (-3, 0, 3))
fis_combined.add_term('prey', 'crescimento', 'triangular', (0, 5, 10))
fis_combined.add_term('prey', 'crescimento_forte', 'triangular', (5, 20, 20))

fis_combined.add_output('predator', (-20, 20))
fis_combined.add_term('predator', 'declinio_forte', 'triangular', (-20, -20, -5))
fis_combined.add_term('predator', 'declinio', 'triangular', (-10, -5, 0))
fis_combined.add_term('predator', 'estavel', 'triangular', (-3, 0, 3))
fis_combined.add_term('predator', 'crescimento', 'triangular', (0, 5, 10))
fis_combined.add_term('predator', 'crescimento_forte', 'triangular', (5, 20, 20))

# Adicionar regras (todas as 9 combinações)
prey_terms = ['baixa', 'media', 'alta']
pred_terms = ['baixa', 'media', 'alta']

prey_outputs = [
    ['crescimento_forte', 'declinio', 'declinio_forte'],
    ['crescimento', 'estavel', 'declinio_forte'],
    ['crescimento', 'estavel', 'declinio']
]

pred_outputs = [
    ['declinio', 'declinio_forte', 'declinio_forte'],
    ['crescimento', 'estavel', 'declinio'],
    ['crescimento_forte', 'crescimento', 'estavel']
]

for i, prey_term in enumerate(prey_terms):
    for j, pred_term in enumerate(pred_terms):
        fis_combined.add_rule(
            {'prey': prey_term, 'predator': pred_term},
            {'prey': prey_outputs[i][j], 'predator': pred_outputs[i][j]}
        )

print(f"✅ FIS Combinado criado:")
print(f"   • {len(fis_combined.rule_base.rules)} regras")
print(f"   • 2 entradas: prey, predator")
print(f"   • 2 saídas: prey, predator")


# ============================================================================
# PARTE 4: Simulação
# ============================================================================

print("\n\n" + "=" * 70)
print("4️⃣  Simulação do Sistema Predador-Presa")
print("=" * 70)

# Criar sistema p-fuzzy contínuo
pfuzzy = fs.dynamic.PFuzzyContinuous(
    fis=fis_combined,
    mode='absolute',
    state_vars=['prey', 'predator'],
    method='rk4'
)

# Condições iniciais
x0 = {'prey': 40, 'predator': 20}

print(f"\n📊 Condições iniciais:")
print(f"   • Presas: {x0['prey']}")
print(f"   • Predadores: {x0['predator']}")

print(f"\n🚀 Simulando de t=0 a t=50...")
trajectory = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 50),
    dt=0.01
)

print(f"✅ Simulação concluída!")
print(f"\n📈 Estatísticas:")
print(f"   Presas:")
print(f"      • Mín: {trajectory[:, 0].min():.2f}")
print(f"      • Máx: {trajectory[:, 0].max():.2f}")
print(f"      • Média: {trajectory[:, 0].mean():.2f}")
print(f"   Predadores:")
print(f"      • Mín: {trajectory[:, 1].min():.2f}")
print(f"      • Máx: {trajectory[:, 1].max():.2f}")
print(f"      • Média: {trajectory[:, 1].mean():.2f}")


# ============================================================================
# PARTE 5: Visualizações
# ============================================================================

print("\n\n" + "=" * 70)
print("5️⃣  Visualizações")
print("=" * 70)

# Criar figura com 3 subplots
fig = plt.figure(figsize=(16, 5))

# Plot 1: Séries temporais
ax1 = plt.subplot(1, 3, 1)
pfuzzy.plot_trajectory(variables=['prey', 'predator'])
plt.title('Dinâmica Temporal', fontsize=14, fontweight='bold')

# Plot 2: Espaço de fase
ax2 = plt.subplot(1, 3, 2)
pfuzzy.plot_phase_space('prey', 'predator',
                        title='Espaço de Fase: Ciclo Predador-Presa')

# Plot 3: Campo vetorial (opcional - mostra direção do fluxo)
ax3 = plt.subplot(1, 3, 3)

# Criar grade para campo vetorial
prey_grid = np.linspace(10, 90, 15)
pred_grid = np.linspace(10, 90, 15)
P, D = np.meshgrid(prey_grid, pred_grid)

# Calcular vetores de campo
dP = np.zeros_like(P)
dD = np.zeros_like(D)

for i in range(len(prey_grid)):
    for j in range(len(pred_grid)):
        state = {'prey': P[j, i], 'predator': D[j, i]}
        result = fis_combined.evaluate(state)
        dP[j, i] = result['prey']
        dD[j, i] = result['predator']

# Plotar campo vetorial
ax3.quiver(P, D, dP, dD, alpha=0.6, color='gray')
ax3.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2,
         label='Trajetória')
ax3.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10,
         label='Inicial')
ax3.set_xlabel('Presas', fontsize=12)
ax3.set_ylabel('Predadores', fontsize=12)
ax3.set_title('Campo Vetorial + Trajetória', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/pfuzzy_predator_prey.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráfico salvo em: /tmp/pfuzzy_predator_prey.png")


# ============================================================================
# PARTE 6: Múltiplas Condições Iniciais
# ============================================================================

print("\n\n" + "=" * 70)
print("6️⃣  Testando Múltiplas Condições Iniciais")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

initial_conditions = [
    {'prey': 30, 'predator': 15},
    {'prey': 50, 'predator': 25},
    {'prey': 70, 'predator': 35},
    {'prey': 40, 'predator': 40}
]

colors = ['blue', 'red', 'green', 'purple']

print("\n🔄 Simulando 4 condições iniciais diferentes...\n")

for i, x0 in enumerate(initial_conditions):
    print(f"{i+1}. Presas={x0['prey']}, Predadores={x0['predator']}")

    traj = pfuzzy.simulate(x0=x0, t_span=(0, 50), dt=0.01, store_all=True)

    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2,
            label=f"({x0['prey']}, {x0['predator']})")
    ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=10)

ax.set_xlabel('Presas', fontsize=12)
ax.set_ylabel('Predadores', fontsize=12)
ax.set_title('Espaço de Fase: Múltiplas Condições Iniciais',
             fontsize=14, fontweight='bold')
ax.legend(title='Condições Iniciais (prey, predator)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/pfuzzy_multiple_ic.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráfico salvo em: /tmp/pfuzzy_multiple_ic.png")


# ============================================================================
# RESUMO
# ============================================================================

print("\n\n" + "=" * 70)
print("RESUMO: Sistema Predador-Presa Fuzzy")
print("=" * 70)

print("""
✅ MODELO IMPLEMENTADO:

Sistema p-fuzzy com 2 variáveis de estado acopladas:
• dx_prey/dt = f_prey(prey, predator)
• dx_pred/dt = f_pred(prey, predator)

Onde f_prey e f_pred são determinados por regras fuzzy.

🎯 CARACTERÍSTICAS:

1. CICLOS OSCILATÓRIOS:
   • Presas crescem → Predadores crescem
   • Predadores crescem → Presas diminuem
   • Predadores diminuem → Presas crescem
   • Ciclo se repete!

2. COMPORTAMENTO QUALITATIVO:
   • Similar ao modelo clássico Lotka-Volterra
   • Mas com regras linguísticas interpretáveis
   • "Se presas são altas E predadores baixos → predadores crescem muito"

3. VANTAGENS DA ABORDAGEM FUZZY:
   • Regras baseadas em conhecimento de especialistas
   • Não requer parâmetros numéricos precisos (α, β, γ, δ)
   • Captura comportamentos não-lineares complexos
   • Fácil de ajustar e interpretar

📊 APLICAÇÕES:

• Ecologia: dinâmica de populações
• Epidemiologia: suscetíveis vs infectados
• Economia: oferta vs demanda
• Química: reações oscilantes
• Engenharia: sistemas de controle acoplados

🔬 EXPERIMENTOS SUGERIDOS:

1. Altere as regras fuzzy e observe os efeitos
2. Teste diferentes funções de pertinência
3. Compare com modelo Lotka-Volterra clássico
4. Adicione termos de competição intraespecífica
5. Explore pontos de equilíbrio
""")

print("=" * 70)
print("✅ Exemplo concluído!")
print("=" * 70)
