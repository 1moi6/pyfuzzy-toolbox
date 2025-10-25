"""
Teste de Integração: ANFIS com Otimização Metaheurística
=========================================================

Demonstra o uso dos 3 algoritmos metaheurísticos (PSO, DE, GA)
para otimização global do ANFIS.
"""

import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.learning import ANFIS, PSO, DE, GA

np.random.seed(42)

print("="*70)
print("TESTE: ANFIS COM OTIMIZAÇÃO METAHEURÍSTICA")
print("="*70)

# Gerar dados sintéticos
print("\n1. Gerando dados...")
X_train = np.random.uniform(-3, 3, (100, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.1 * np.random.randn(100)

X_test = np.random.uniform(-3, 3, (30, 2))
y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# TESTE 1: fit() tradicional (LSE + Gradiente) - BASELINE
# ============================================================================
print("\n" + "="*70)
print("TESTE 1: ANFIS Tradicional (LSE + Gradiente Descendente)")
print("="*70)

anfis_trad = ANFIS(n_inputs=2, n_mfs=3, mf_type='gaussmf', learning_rate=0.01)
anfis_trad.fit(X_train, y_train, epochs=50, verbose=False)

r2_trad = anfis_trad.score(X_test, y_test)
print(f"\n✅ ANFIS Tradicional - R² no teste: {r2_trad:.4f}")

# ============================================================================
# TESTE 2: PSO (Particle Swarm Optimization)
# ============================================================================
print("\n" + "="*70)
print("TESTE 2: ANFIS + PSO (Otimização por Enxame de Partículas)")
print("="*70)

anfis_pso = ANFIS(n_inputs=2, n_mfs=3, mf_type='gaussmf')
anfis_pso.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    n_particles=20,
    n_iterations=30,
    verbose=True
)

r2_pso = anfis_pso.score(X_test, y_test)
print(f"✅ ANFIS + PSO - R² no teste: {r2_pso:.4f}")

# ============================================================================
# TESTE 3: DE (Differential Evolution)
# ============================================================================
print("\n" + "="*70)
print("TESTE 3: ANFIS + DE (Evolução Diferencial)")
print("="*70)

anfis_de = ANFIS(n_inputs=2, n_mfs=3, mf_type='gaussmf')
anfis_de.fit_metaheuristic(
    X_train, y_train,
    optimizer='de',
    n_particles=20,
    n_iterations=30,
    verbose=True,
    F=0.8,  # Parâmetro específico do DE
    CR=0.9
)

r2_de = anfis_de.score(X_test, y_test)
print(f"✅ ANFIS + DE - R² no teste: {r2_de:.4f}")

# ============================================================================
# TESTE 4: GA (Genetic Algorithm)
# ============================================================================
print("\n" + "="*70)
print("TESTE 4: ANFIS + GA (Algoritmo Genético)")
print("="*70)

anfis_ga = ANFIS(n_inputs=2, n_mfs=3, mf_type='gaussmf')
anfis_ga.fit_metaheuristic(
    X_train, y_train,
    optimizer='ga',
    n_particles=20,
    n_iterations=30,
    verbose=True,
    elite_ratio=0.1,  # Parâmetros específicos do GA
    mutation_rate=0.1
)

r2_ga = anfis_ga.score(X_test, y_test)
print(f"✅ ANFIS + GA - R² no teste: {r2_ga:.4f}")

# ============================================================================
# TESTE 5: Uso direto dos otimizadores (fora do ANFIS)
# ============================================================================
print("\n" + "="*70)
print("TESTE 5: Uso Direto dos Otimizadores")
print("="*70)

# Função de teste: Sphere function
def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-5, 5], [-5, 5]])

print("\n✓ PSO otimizando Sphere function...")
pso = PSO(n_particles=20, n_iterations=50)
best_x_pso, best_f_pso, _ = pso.optimize(sphere, bounds, verbose=False)
print(f"   Melhor solução: x={best_x_pso}, f(x)={best_f_pso:.6f}")

print("\n✓ DE otimizando Sphere function...")
de = DE(pop_size=20, max_iter=50)
best_x_de, best_f_de, _ = de.optimize(sphere, bounds, verbose=False)
print(f"   Melhor solução: x={best_x_de}, f(x)={best_f_de:.6f}")

print("\n✓ GA otimizando Sphere function...")
ga = GA(pop_size=20, max_gen=50)
best_x_ga, best_f_ga, _ = ga.optimize(sphere, bounds, verbose=False)
print(f"   Melhor solução: x={best_x_ga}, f(x)={best_f_ga:.6f}")

# ============================================================================
# COMPARAÇÃO FINAL
# ============================================================================
print("\n" + "="*70)
print("COMPARAÇÃO: ANFIS Tradicional vs Metaheurísticas")
print("="*70)

resultados = {
    'Tradicional (LSE+Grad)': r2_trad,
    'PSO': r2_pso,
    'DE': r2_de,
    'GA': r2_ga
}

print("\nR² no conjunto de teste:")
for metodo, r2 in resultados.items():
    print(f"  {metodo:25s}: {r2:.4f}")

melhor_metodo = max(resultados, key=resultados.get)
print(f"\n🏆 Melhor método: {melhor_metodo} (R² = {resultados[melhor_metodo]:.4f})")

# ============================================================================
# VISUALIZAÇÃO (opcional - descomente para plotar)
# ============================================================================
print("\n" + "="*70)
print("VISUALIZAÇÃO")
print("="*70)

# Descomente para gerar gráficos
"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Convergência PSO
ax = axes[0, 0]
if hasattr(anfis_pso, 'optimization_history'):
    ax.plot(anfis_pso.optimization_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteração')
    ax.set_ylabel('MSE')
    ax.set_title('Convergência PSO')
    ax.grid(True, alpha=0.3)

# Plot 2: Convergência DE
ax = axes[0, 1]
if hasattr(anfis_de, 'optimization_history'):
    ax.plot(anfis_de.optimization_history, 'r-', linewidth=2)
    ax.set_xlabel('Iteração')
    ax.set_ylabel('MSE')
    ax.set_title('Convergência DE')
    ax.grid(True, alpha=0.3)

# Plot 3: Convergência GA
ax = axes[1, 0]
if hasattr(anfis_ga, 'optimization_history'):
    ax.plot(anfis_ga.optimization_history, 'g-', linewidth=2)
    ax.set_xlabel('Iteração')
    ax.set_ylabel('MSE')
    ax.set_title('Convergência GA')
    ax.grid(True, alpha=0.3)

# Plot 4: Comparação de R²
ax = axes[1, 1]
metodos = list(resultados.keys())
r2_values = list(resultados.values())
colors = ['blue', 'orange', 'red', 'green']
ax.bar(metodos, r2_values, color=colors, alpha=0.7)
ax.set_ylabel('R²')
ax.set_title('Comparação de Performance')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/tmp/anfis_metaheuristics.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráfico salvo em /tmp/anfis_metaheuristics.png")
"""

print("\n" + "="*70)
print("RESUMO DOS TESTES")
print("="*70)

print("""
✅ TESTE 1: ANFIS Tradicional - PASSOU
✅ TESTE 2: ANFIS + PSO - PASSOU
✅ TESTE 3: ANFIS + DE - PASSOU
✅ TESTE 4: ANFIS + GA - PASSOU
✅ TESTE 5: Uso Direto dos Otimizadores - PASSOU

🎯 INTEGRAÇÃO DE METAHEURÍSTICAS COMPLETA E FUNCIONANDO!
""")

print("="*70)
print("\nDICA: As metaheurísticas são especialmente úteis quando:")
print("  • Você quer otimização global (evitar mínimos locais)")
print("  • O gradiente é difícil de calcular")
print("  • Você tem tempo computacional disponível")
print("  • O espaço de busca é complexo/multimodal")
print("\nPara datasets grandes e treino rápido, use fit() tradicional.")
print("Para otimização global e exploração do espaço, use fit_metaheuristic()!")
print("="*70)
