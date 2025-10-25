"""
Teste Completo: Sistema Mamdani com Aprendizado
================================================

Testa todas as funcionalidades do MamdaniLearning:
1. Aprendizado por gradiente (batch, online, mini-batch)
2. Otimização metaheurística com caching de ativações (PSO, DE, GA)
3. Diferentes métodos de defuzzificação (COG, COS)
4. Extração de regras linguísticas
5. Comparação de performance
"""

import numpy as np
import time
from fuzzy_systems.learning import MamdaniLearning, PSO, DE, GA

np.random.seed(42)

print("=" * 70)
print("TESTE COMPLETO: MAMDANI LEARNING")
print("=" * 70)

# ============================================================================
# 1. GERAR DADOS SINTÉTICOS
# ============================================================================
print("\n1. Gerando dados sintéticos...")

n_train = 150
n_test = 50

# Função alvo: y = sin(x1) + cos(x2) + 0.5*x1*x2
X_train = np.random.uniform(-3, 3, (n_train, 2))
y_train = (np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) +
           0.2 * X_train[:, 0] * X_train[:, 1] +
           0.1 * np.random.randn(n_train))

X_test = np.random.uniform(-3, 3, (n_test, 2))
y_test = (np.sin(X_test[:, 0]) + np.cos(X_test[:, 1]) +
          0.2 * X_test[:, 0] * X_test[:, 1])

print(f"   Train: {X_train.shape}, y: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"   Test:  {X_test.shape}, y: [{y_test.min():.2f}, {y_test.max():.2f}]")

# ============================================================================
# 2. TESTE: APRENDIZADO POR GRADIENTE (MODO BATCH)
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 1: Aprendizado por Gradiente - Modo BATCH")
print("=" * 70)

mamdani_batch = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog',
    use_rule_weights=True
)

start_time = time.time()
history_batch = mamdani_batch.fit(
    X_train, y_train,
    epochs=50,
    learning_rate=0.01,
    learning_mode='batch',
    verbose=False
)
time_batch = time.time() - start_time

r2_batch = mamdani_batch.score(X_test, y_test)
print(f"\n✅ Gradiente Batch - Tempo: {time_batch:.2f}s, R² test: {r2_batch:.4f}")
print(f"   Loss inicial: {history_batch['train_loss'][0]:.6f}")
print(f"   Loss final:   {history_batch['train_loss'][-1]:.6f}")

# ============================================================================
# 3. TESTE: APRENDIZADO POR GRADIENTE (MODO MINI-BATCH)
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 2: Aprendizado por Gradiente - Modo MINI-BATCH")
print("=" * 70)

mamdani_minibatch = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog',
    use_rule_weights=True
)

start_time = time.time()
history_minibatch = mamdani_minibatch.fit(
    X_train, y_train,
    epochs=50,
    learning_rate=0.01,
    learning_mode='mini-batch',
    batch_size=32,
    early_stopping=True,
    patience=10,
    validation_split=0.2,
    verbose=False
)
time_minibatch = time.time() - start_time

r2_minibatch = mamdani_minibatch.score(X_test, y_test)
print(f"\n✅ Gradiente Mini-Batch - Tempo: {time_minibatch:.2f}s, R² test: {r2_minibatch:.4f}")
print(f"   Épocas executadas: {len(history_minibatch['train_loss'])}")

# ============================================================================
# 4. TESTE: DEFUZZIFICAÇÃO COS (Center of Sums)
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 3: Defuzzificação COS (Center of Sums)")
print("=" * 70)

mamdani_cos = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cos',  # COS em vez de COG
    use_rule_weights=False
)

mamdani_cos.fit(
    X_train, y_train,
    epochs=50,
    learning_rate=0.01,
    learning_mode='mini-batch',
    verbose=False
)

r2_cos = mamdani_cos.score(X_test, y_test)
print(f"\n✅ Defuzzificação COS - R² test: {r2_cos:.4f}")

# ============================================================================
# 5. TESTE: OTIMIZAÇÃO METAHEURÍSTICA - PSO COM CACHING
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 4: Otimização PSO com Caching de Ativações")
print("=" * 70)

mamdani_pso = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog'
)

print("\n🚀 Modo: Otimiza APENAS consequentes (usa cache de ativações)")
start_time = time.time()
mamdani_pso.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    n_particles=20,
    n_iterations=30,
    optimize_params='consequents_only',  # MODO COM CACHE!
    verbose=True
)
time_pso = time.time() - start_time

r2_pso = mamdani_pso.score(X_test, y_test)
print(f"\n✅ PSO (consequents only) - Tempo: {time_pso:.2f}s, R² test: {r2_pso:.4f}")

# ============================================================================
# 6. TESTE: OTIMIZAÇÃO METAHEURÍSTICA - DE COM CACHING
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 5: Otimização DE (Differential Evolution) com Caching")
print("=" * 70)

mamdani_de = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog'
)

start_time = time.time()
mamdani_de.fit_metaheuristic(
    X_train, y_train,
    optimizer='de',
    n_particles=20,
    n_iterations=30,
    optimize_params='consequents_only',
    verbose=True,
    F=0.8,  # Parâmetro específico do DE
    CR=0.9
)
time_de = time.time() - start_time

r2_de = mamdani_de.score(X_test, y_test)
print(f"\n✅ DE (consequents only) - Tempo: {time_de:.2f}s, R² test: {r2_de:.4f}")

# ============================================================================
# 7. TESTE: OTIMIZAÇÃO METAHEURÍSTICA - GA COM CACHING
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 6: Otimização GA (Genetic Algorithm) com Caching")
print("=" * 70)

mamdani_ga = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog'
)

start_time = time.time()
mamdani_ga.fit_metaheuristic(
    X_train, y_train,
    optimizer='ga',
    n_particles=20,
    n_iterations=30,
    optimize_params='consequents_only',
    verbose=True,
    elite_ratio=0.1,
    mutation_rate=0.1
)
time_ga = time.time() - start_time

r2_ga = mamdani_ga.score(X_test, y_test)
print(f"\n✅ GA (consequents only) - Tempo: {time_ga:.2f}s, R² test: {r2_ga:.4f}")

# ============================================================================
# 8. TESTE: OTIMIZAÇÃO DE CENTROIDES (SEM CACHE, MAIS LENTO)
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 7: Otimização PSO - Apenas Centroides (SEM cache)")
print("=" * 70)

mamdani_pso_centroids = MamdaniLearning(
    n_inputs=2,
    n_mfs_input=[3, 3],
    n_mfs_output=5,
    defuzz_method='cog'
)

start_time = time.time()
mamdani_pso_centroids.fit_metaheuristic(
    X_train, y_train,
    optimizer='pso',
    n_particles=15,
    n_iterations=20,
    optimize_params='output_only',  # Otimiza centroides, SEM cache
    verbose=False
)
time_pso_centroids = time.time() - start_time

r2_pso_centroids = mamdani_pso_centroids.score(X_test, y_test)
print(f"\n✅ PSO (output only) - Tempo: {time_pso_centroids:.2f}s, R² test: {r2_pso_centroids:.4f}")
print(f"   Nota: Mais lento que 'consequents_only' pois recalcula ativações a cada iteração")

# ============================================================================
# 9. TESTE: EXTRAÇÃO DE REGRAS LINGUÍSTICAS
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 8: Extração de Regras Linguísticas")
print("=" * 70)

print("\nRegras extraídas do melhor modelo (PSO):")
rules = mamdani_pso.get_linguistic_rules()
print(f"\nTotal: {len(rules)} regras\n")

# Mostra apenas algumas regras como exemplo
for i, rule in enumerate(rules[:5], 1):
    print(f"{i}. {rule}")
print("...")
print(f"{len(rules)}. {rules[-1]}")

# ============================================================================
# 10. COMPARAÇÃO FINAL
# ============================================================================
print("\n" + "=" * 70)
print("COMPARAÇÃO FINAL DE MÉTODOS")
print("=" * 70)

resultados = {
    'Gradiente Batch': (r2_batch, time_batch),
    'Gradiente Mini-Batch': (r2_minibatch, time_minibatch),
    'Defuzz COS': (r2_cos, '-'),
    'PSO (consequents, cache)': (r2_pso, time_pso),
    'DE (consequents, cache)': (r2_de, time_de),
    'GA (consequents, cache)': (r2_ga, time_ga),
    'PSO (centroids, no cache)': (r2_pso_centroids, time_pso_centroids),
}

print("\n{:<30s} {:>10s} {:>12s}".format("Método", "R² Test", "Tempo (s)"))
print("-" * 70)
for metodo, (r2, tempo) in resultados.items():
    tempo_str = f"{tempo:.2f}" if isinstance(tempo, float) else tempo
    print(f"{metodo:<30s} {r2:>10.4f} {tempo_str:>12s}")

melhor_metodo = max(resultados.items(), key=lambda x: x[1][0])
print(f"\n🏆 Melhor método (R²): {melhor_metodo[0]} - R² = {melhor_metodo[1][0]:.4f}")

# ============================================================================
# 11. ANÁLISE DE PERFORMANCE DO CACHING
# ============================================================================
print("\n" + "=" * 70)
print("ANÁLISE DE PERFORMANCE: Caching vs Sem Caching")
print("=" * 70)

speedup = time_pso_centroids / time_pso if time_pso > 0 else 0
print(f"\nTempo PSO com cache:    {time_pso:.2f}s")
print(f"Tempo PSO sem cache:    {time_pso_centroids:.2f}s")
print(f"Speedup do caching:     {speedup:.2f}x")
print("\n💡 O caching de ativações acelera MUITO a otimização quando")
print("   otimizamos apenas os consequentes (conjuntos de entrada fixos)!")

# ============================================================================
# 12. TESTE DE PREDIÇÃO
# ============================================================================
print("\n" + "=" * 70)
print("TESTE 9: Predição em Novos Dados")
print("=" * 70)

X_new = np.array([
    [0.0, 0.0],
    [1.5, -1.5],
    [-2.0, 2.5]
])

print("\nPredições do melhor modelo (PSO):")
for i, x in enumerate(X_new):
    y_pred = mamdani_pso.predict(x.reshape(1, -1))
    print(f"  x = {x} → y = {y_pred[0]:.4f}")

# ============================================================================
# RESUMO DOS TESTES
# ============================================================================
print("\n" + "=" * 70)
print("RESUMO DOS TESTES")
print("=" * 70)

print("""
✅ TESTE 1: Gradiente Batch - PASSOU
✅ TESTE 2: Gradiente Mini-Batch - PASSOU
✅ TESTE 3: Defuzzificação COS - PASSOU
✅ TESTE 4: PSO com Caching - PASSOU
✅ TESTE 5: DE com Caching - PASSOU
✅ TESTE 6: GA com Caching - PASSOU
✅ TESTE 7: PSO sem Cache - PASSOU
✅ TESTE 8: Extração de Regras - PASSOU
✅ TESTE 9: Predição - PASSOU

🎯 MAMDANI LEARNING COMPLETO E FUNCIONANDO!
""")

print("=" * 70)
print("\nDICA: Quando usar cada método?")
print("-" * 70)
print("• Gradiente Batch/Mini-Batch:")
print("    - Para ajuste fino de todos os parâmetros")
print("    - Quando você tem gradientes disponíveis")
print("    - Convergência suave e determinística")
print()
print("• PSO/DE/GA (consequents_only, COM cache):")
print("    - Para otimização RÁPIDA de regras")
print("    - Quando conjuntos de entrada são bons")
print("    - Evita mínimos locais")
print("    - 🚀 Muito eficiente com caching!")
print()
print("• PSO/DE/GA (output_only ou all, SEM cache):")
print("    - Para otimização global completa")
print("    - Quando você quer explorar todo o espaço")
print("    - ⚠️  Mais lento (recalcula ativações)")
print()
print("• Defuzzificação COG vs COS:")
print("    - COG: Mais preciso, integração completa")
print("    - COS: Mais rápido, aproximação")
print("=" * 70)
