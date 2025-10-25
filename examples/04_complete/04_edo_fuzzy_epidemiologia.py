"""
Exemplo Completo: Modelo Epidemiológico com Incerteza Fuzzy (SIR)
==================================================================

Demonstra o uso completo do solver de EDO fuzzy:
1. Modelo SIR clássico (Suscetíveis-Infectados-Recuperados)
2. Parâmetros com incerteza fuzzy (taxas de transmissão/recuperação)
3. Solução e propagação de incerteza
4. Visualização de envelopes fuzzy
5. Análise de cenários
6. Exportação de resultados

Modelo SIR:
    dS/dt = -β * S * I / N
    dI/dt = β * S * I / N - γ * I
    dR/dt = γ * I

onde:
    S = Suscetíveis
    I = Infectados
    R = Recuperados
    β = Taxa de transmissão (incerta)
    γ = Taxa de recuperação (incerta)
    N = População total (constante)
"""

import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.dynamics import FuzzyNumber, FuzzyODESolver

print("="*70)
print("MODELO EPIDEMIOLÓGICO SIR COM INCERTEZA FUZZY")
print("="*70)

# ============================================================================
# 1. DEFINIR MODELO SIR
# ============================================================================
print("\n1. Definindo modelo epidemiológico SIR...")

def sir_model(t, y, beta, gamma, N):
    """
    Modelo SIR clássico

    Args:
        t: Tempo
        y: [S, I, R] - Estado atual
        beta: Taxa de transmissão
        gamma: Taxa de recuperação
        N: População total

    Returns:
        [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return np.array([dS, dI, dR])

print("   ✓ Modelo SIR definido")
print("   - dS/dt = -β*S*I/N")
print("   - dI/dt = β*S*I/N - γ*I")
print("   - dR/dt = γ*I")

# ============================================================================
# 2. DEFINIR PARÂMETROS COM INCERTEZA
# ============================================================================
print("\n2. Definindo parâmetros com incerteza...")

# População
N = 1000  # População total (constante)

# Condições iniciais (com incerteza na estimativa inicial de infectados)
S0_crisp = 990  # Suscetíveis inicial (certo)
I0_fuzzy = FuzzyNumber.triangular(center=10, spread=3, name="I(0)")  # Infectados inicial (incerto)
R0_crisp = 0    # Recuperados inicial (certo)

print(f"   ✓ População total: N = {N}")
print(f"   ✓ S(0) = {S0_crisp} (certo)")
print(f"   ✓ I(0) ~ {I0_fuzzy.fuzzy_set.params[1]} ± {I0_fuzzy.spread} (fuzzy)")
print(f"   ✓ R(0) = {R0_crisp} (certo)")

# Parâmetros epidemiológicos (incertos)
beta_fuzzy = FuzzyNumber.triangular(
    center=0.5,   # Taxa de transmissão ~0.5 por dia
    spread=0.1,   # Incerteza ±0.1
    name="beta"
)

gamma_fuzzy = FuzzyNumber.triangular(
    center=0.1,   # Taxa de recuperação ~0.1 por dia (período de ~10 dias)
    spread=0.02,  # Incerteza ±0.02
    name="gamma"
)

print(f"   ✓ β ~ {beta_fuzzy.fuzzy_set.params[1]} ± {beta_fuzzy.spread} por dia (fuzzy)")
print(f"   ✓ γ ~ {gamma_fuzzy.fuzzy_set.params[1]} ± {gamma_fuzzy.spread} por dia (fuzzy)")

# Número básico de reprodução R0
R0_center = beta_fuzzy.fuzzy_set.params[1] / gamma_fuzzy.fuzzy_set.params[1]
print(f"   ✓ R₀ ~ {R0_center:.2f} (número básico de reprodução)")

# ============================================================================
# 3. CONFIGURAR SOLVER DE EDO FUZZY
# ============================================================================
print("\n3. Configurando solver de EDO fuzzy...")

solver = FuzzyODESolver(
    ode_func=sir_model,
    t_span=(0, 100),  # 100 dias
    y0_fuzzy=[S0_crisp, I0_fuzzy, R0_crisp],
    params={
        'beta': beta_fuzzy,
        'gamma': gamma_fuzzy,
        'N': N  # População constante (crisp)
    },
    n_alpha_cuts=11,    # 11 níveis-α
    n_grid_points=5,    # 5 pontos por dimensão
    method='RK45',
    t_eval=np.linspace(0, 100, 200),
    var_names=['Suscetíveis', 'Infectados', 'Recuperados']
)

print(f"   ✓ Solver configurado")
print(f"   - Método: {solver.method}")
print(f"   - α-níveis: {solver.n_alpha_cuts}")
print(f"   - Grid points: {solver.n_grid_points}")
print(f"   - Período: {solver.t_span[0]}-{solver.t_span[1]} dias")

# ============================================================================
# 4. RESOLVER SISTEMA FUZZY
# ============================================================================
print("\n4. Resolvendo sistema com propagação de incerteza...")
print("   (Pode levar alguns segundos...)")

sol = solver.solve(verbose=True)

print("\n   ✓ Solução computada!")
print(f"   - {len(sol.t)} pontos temporais")
print(f"   - {len(sol.alphas)} níveis-α")
print(f"   - {sol.y_min.shape[1]} variáveis")

# ============================================================================
# 5. ANÁLISE DE RESULTADOS
# ============================================================================
print("\n5. Analisando resultados...")

# Pico de infectados
I_max_alpha1 = sol.y_max[-1, 1, :].max()  # α=1 (núcleo)
I_min_alpha1 = sol.y_min[-1, 1, :].max()
t_peak_max = sol.t[sol.y_max[-1, 1, :].argmax()]

# Final da epidemia (α=1)
S_final = sol.y_min[-1, 0, -1]
I_final = sol.y_max[-1, 1, -1]
R_final = sol.y_max[-1, 2, -1]

print(f"\n   Pico de infectados (α=1.0):")
print(f"   - Tempo: {t_peak_max:.1f} dias")
print(f"   - Infectados: [{I_min_alpha1:.0f}, {I_max_alpha1:.0f}]")
print(f"   - % população: [{I_min_alpha1/N*100:.1f}%, {I_max_alpha1/N*100:.1f}%]")

print(f"\n   Estado final (t=100 dias, α=1.0):")
print(f"   - Suscetíveis: {S_final:.0f} ({S_final/N*100:.1f}%)")
print(f"   - Infectados: {I_final:.0f} ({I_final/N*100:.1f}%)")
print(f"   - Recuperados: {R_final:.0f} ({R_final/N*100:.1f}%)")

# ============================================================================
# 6. VISUALIZAÇÃO
# ============================================================================
print("\n6. Criando visualizações...")

fig = plt.figure(figsize=(18, 10))

# 6.1 Suscetíveis
ax1 = plt.subplot(2, 3, 1)
for i, alpha in enumerate([0, 0.5, 1.0]):
    idx = np.argmin(np.abs(sol.alphas - alpha))
    S_min = sol.y_min[idx, 0, :]
    S_max = sol.y_max[idx, 0, :]
    color = plt.cm.Blues(0.3 + 0.7 * alpha)
    ax1.fill_between(sol.t, S_min, S_max, alpha=0.4, color=color,
                     label=f'α={alpha:.1f}')
ax1.set_xlabel('Tempo (dias)', fontsize=11)
ax1.set_ylabel('Número de indivíduos', fontsize=11)
ax1.set_title('Suscetíveis (S)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6.2 Infectados
ax2 = plt.subplot(2, 3, 2)
for i, alpha in enumerate([0, 0.5, 1.0]):
    idx = np.argmin(np.abs(sol.alphas - alpha))
    I_min = sol.y_min[idx, 1, :]
    I_max = sol.y_max[idx, 1, :]
    color = plt.cm.Reds(0.3 + 0.7 * alpha)
    ax2.fill_between(sol.t, I_min, I_max, alpha=0.4, color=color,
                     label=f'α={alpha:.1f}')
ax2.set_xlabel('Tempo (dias)', fontsize=11)
ax2.set_ylabel('Número de indivíduos', fontsize=11)
ax2.set_title('Infectados (I)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 Recuperados
ax3 = plt.subplot(2, 3, 3)
for i, alpha in enumerate([0, 0.5, 1.0]):
    idx = np.argmin(np.abs(sol.alphas - alpha))
    R_min = sol.y_min[idx, 2, :]
    R_max = sol.y_max[idx, 2, :]
    color = plt.cm.Greens(0.3 + 0.7 * alpha)
    ax3.fill_between(sol.t, R_min, R_max, alpha=0.4, color=color,
                     label=f'α={alpha:.1f}')
ax3.set_xlabel('Tempo (dias)', fontsize=11)
ax3.set_ylabel('Número de indivíduos', fontsize=11)
ax3.set_title('Recuperados (R)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 6.4 Todas as variáveis juntas (α=1.0)
ax4 = plt.subplot(2, 3, 4)
idx_alpha1 = -1  # α=1.0 é o último
ax4.fill_between(sol.t, sol.y_min[idx_alpha1, 0, :], sol.y_max[idx_alpha1, 0, :],
                 alpha=0.4, color='blue', label='Suscetíveis')
ax4.fill_between(sol.t, sol.y_min[idx_alpha1, 1, :], sol.y_max[idx_alpha1, 1, :],
                 alpha=0.4, color='red', label='Infectados')
ax4.fill_between(sol.t, sol.y_min[idx_alpha1, 2, :], sol.y_max[idx_alpha1, 2, :],
                 alpha=0.4, color='green', label='Recuperados')
ax4.set_xlabel('Tempo (dias)', fontsize=11)
ax4.set_ylabel('Número de indivíduos', fontsize=11)
ax4.set_title('Modelo SIR Completo (α=1.0)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 6.5 Evolução da largura dos envelopes (incerteza)
ax5 = plt.subplot(2, 3, 5)
largura_S = sol.y_max[-1, 0, :] - sol.y_min[-1, 0, :]
largura_I = sol.y_max[-1, 1, :] - sol.y_min[-1, 1, :]
largura_R = sol.y_max[-1, 2, :] - sol.y_min[-1, 2, :]
ax5.plot(sol.t, largura_S, label='Suscetíveis', linewidth=2)
ax5.plot(sol.t, largura_I, label='Infectados', linewidth=2)
ax5.plot(sol.t, largura_R, label='Recuperados', linewidth=2)
ax5.set_xlabel('Tempo (dias)', fontsize=11)
ax5.set_ylabel('Largura do envelope (α=1.0)', fontsize=11)
ax5.set_title('Propagação de Incerteza', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6.6 % da população (α=1.0)
ax6 = plt.subplot(2, 3, 6)
ax6.fill_between(sol.t, sol.y_min[idx_alpha1, 0, :]/N*100, sol.y_max[idx_alpha1, 0, :]/N*100,
                 alpha=0.4, color='blue', label='Suscetíveis')
ax6.fill_between(sol.t, sol.y_min[idx_alpha1, 1, :]/N*100, sol.y_max[idx_alpha1, 1, :]/N*100,
                 alpha=0.4, color='red', label='Infectados')
ax6.fill_between(sol.t, sol.y_min[idx_alpha1, 2, :]/N*100, sol.y_max[idx_alpha1, 2, :]/N*100,
                 alpha=0.4, color='green', label='Recuperados')
ax6.set_xlabel('Tempo (dias)', fontsize=11)
ax6.set_ylabel('% da População', fontsize=11)
ax6.set_title('Modelo SIR em % (α=1.0)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('/tmp/sir_fuzzy.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráficos salvos em /tmp/sir_fuzzy.png")

# ============================================================================
# 7. EXPORTAR RESULTADOS
# ============================================================================
print("\n7. Exportando resultados...")

# Para DataFrame
df = sol.to_dataframe(alpha=1.0)
print(f"   ✓ DataFrame criado ({df.shape[0]} linhas)")

# Para CSV
sol.to_csv('/tmp/sir_fuzzy_alpha1.csv', alpha=1.0)
print("   ✓ CSV salvo (α=1.0): /tmp/sir_fuzzy_alpha1.csv")

sol.to_csv('/tmp/sir_fuzzy_alpha0.csv', alpha=0.0)
print("   ✓ CSV salvo (α=0.0): /tmp/sir_fuzzy_alpha0.csv")

# ============================================================================
# 8. RESUMO
# ============================================================================
print("\n" + "="*70)
print("RESUMO DA SIMULAÇÃO")
print("="*70)
print(f"""
Modelo: SIR com incerteza fuzzy

Parâmetros:
• População: N = {N}
• I(0): {I0_fuzzy.fuzzy_set.params[1]:.0f} ± {I0_fuzzy.spread:.0f} (fuzzy)
• β: {beta_fuzzy.fuzzy_set.params[1]:.2f} ± {beta_fuzzy.spread:.2f} por dia (fuzzy)
• γ: {gamma_fuzzy.fuzzy_set.params[1]:.2f} ± {gamma_fuzzy.spread:.2f} por dia (fuzzy)
• R₀: ~{R0_center:.2f}

Resultados (α=1.0):
• Pico de infectados: {t_peak_max:.0f} dias
• Máximo de infectados: {I_max_alpha1:.0f} ({I_max_alpha1/N*100:.1f}%)
• Estado final (100 dias):
  - Suscetíveis: {S_final:.0f} ({S_final/N*100:.1f}%)
  - Recuperados: {R_final:.0f} ({R_final/N*100:.1f}%)

Insights:
✓ Incerteza nos parâmetros propaga ao longo do tempo
✓ Envelopes fuzzy capturam cenários possíveis
✓ Análise de pior/melhor caso facilitada
✓ Resultados exportados para análise adicional

Aplicações:
• Planejamento de saúde pública com incerteza
• Análise de sensibilidade automática
• Suporte à tomada de decisão em epidemias
• Comunicação de incerteza para tomadores de decisão
""")

print("="*70)
plt.show()
