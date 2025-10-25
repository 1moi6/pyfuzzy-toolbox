"""
Exemplo Completo: ANFIS para Aproximação de Função
===================================================

Demonstra o uso completo do ANFIS:
1. Geração de dados de treinamento/teste
2. Configuração e treinamento do ANFIS
3. Avaliação de performance
4. Visualização de resultados
5. Análise de convergência

Problema: Aproximar a função não-linear:
    f(x, y) = sin(x) * cos(y) + 0.1*x*y
"""

import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.learning import ANFIS

np.random.seed(42)

print("="*70)
print("ANFIS - APROXIMAÇÃO DE FUNÇÃO NÃO-LINEAR")
print("="*70)

# ============================================================================
# 1. GERAR DADOS
# ============================================================================
print("\n1. Gerando dados de treinamento e teste...")

# Função alvo
def target_function(x, y):
    return np.sin(x) * np.cos(y) + 0.1 * x * y

# Dados de treinamento (grid regular + ruído)
n_train = 200
X_train = np.random.uniform([-3, -3], [3, 3], (n_train, 2))
y_train = target_function(X_train[:, 0], X_train[:, 1])
y_train += np.random.normal(0, 0.05, n_train)  # Ruído

# Dados de teste (grid fino)
n_test_per_dim = 30
x_test = np.linspace(-3, 3, n_test_per_dim)
y_test_grid = np.linspace(-3, 3, n_test_per_dim)
X_test_grid, Y_test_grid = np.meshgrid(x_test, y_test_grid)
X_test = np.column_stack([X_test_grid.ravel(), Y_test_grid.ravel()])
y_test = target_function(X_test[:, 0], X_test[:, 1])

print(f"   ✓ {n_train} amostras de treinamento")
print(f"   ✓ {len(X_test)} pontos de teste ({n_test_per_dim}x{n_test_per_dim} grid)")

# ============================================================================
# 2. CONFIGURAR ANFIS
# ============================================================================
print("\n2. Configurando ANFIS...")

anfis = ANFIS(
    n_inputs=2,
    n_rules=16,  # 4x4 grid de regras
    mf_type='gaussian',
    consequent_type='linear',
    regularization='l2',
    lambda_reg=0.001
)

print(f"   ✓ Arquitetura: {anfis.n_inputs} entradas, {anfis.n_rules} regras")
print(f"   ✓ MFs: {anfis.mf_type}")
print(f"   ✓ Consequentes: {anfis.consequent_type}")
print(f"   ✓ Regularização: {anfis.regularization} (λ={anfis.lambda_reg})")

# ============================================================================
# 3. TREINAR ANFIS
# ============================================================================
print("\n3. Treinando ANFIS...")

history = anfis.fit(
    X_train, y_train,
    epochs=200,
    learning_rate=0.01,
    adaptive_lr=True,  # Lyapunov stability
    patience=20,
    verbose=True
)

print("\n   ✓ Treinamento concluído!")
print(f"   - Épocas: {len(history['train_loss'])}")
print(f"   - RMSE final (treino): {history['train_loss'][-1]:.6f}")
if 'val_loss' in history:
    print(f"   - RMSE final (validação): {history['val_loss'][-1]:.6f}")

# ============================================================================
# 4. AVALIAR PERFORMANCE
# ============================================================================
print("\n4. Avaliando performance...")

# Predição
y_pred_train = anfis.predict(X_train)
y_pred_test = anfis.predict(X_test)

# Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n   Métricas de Performance:")
print(f"   ------------------------")
print(f"   RMSE (treino): {rmse_train:.6f}")
print(f"   RMSE (teste):  {rmse_test:.6f}")
print(f"   MAE (teste):   {mae_test:.6f}")
print(f"   R² (teste):    {r2_test:.6f}")

# Score sklearn
score = anfis.score(X_test, y_test)
print(f"   Score (R²):    {score:.6f}")

# ============================================================================
# 5. VISUALIZAÇÃO
# ============================================================================
print("\n5. Criando visualizações...")

fig = plt.figure(figsize=(18, 10))

# 5.1 Convergência
ax1 = plt.subplot(2, 4, 1)
ax1.semilogy(history['train_loss'], label='Treino', linewidth=2)
if 'val_loss' in history:
    ax1.semilogy(history['val_loss'], label='Validação', linewidth=2)
ax1.set_xlabel('Época', fontsize=11)
ax1.set_ylabel('RMSE (log)', fontsize=11)
ax1.set_title('Curva de Convergência', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2 Taxa de aprendizado adaptativa
if 'learning_rate' in history:
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(history['learning_rate'], linewidth=2, color='orange')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('Taxa de Aprendizado', fontsize=11)
    ax2.set_title('Taxa de Aprendizado Adaptativa\n(Lyapunov)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

# 5.3 Predição vs Real (Treino)
ax3 = plt.subplot(2, 4, 3)
ax3.scatter(y_train, y_pred_train, alpha=0.5, s=20)
lim = [min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())]
ax3.plot(lim, lim, 'r--', linewidth=2, label='Ideal')
ax3.set_xlabel('Valor Real', fontsize=11)
ax3.set_ylabel('Valor Predito', fontsize=11)
ax3.set_title(f'Predição vs Real (Treino)\nR²={r2_score(y_train, y_pred_train):.4f}',
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 Predição vs Real (Teste)
ax4 = plt.subplot(2, 4, 4)
ax4.scatter(y_test, y_pred_test, alpha=0.5, s=20, color='green')
lim = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax4.plot(lim, lim, 'r--', linewidth=2, label='Ideal')
ax4.set_xlabel('Valor Real', fontsize=11)
ax4.set_ylabel('Valor Predito', fontsize=11)
ax4.set_title(f'Predição vs Real (Teste)\nR²={r2_test:.4f}',
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5.5 Superfície real
ax5 = plt.subplot(2, 4, 5, projection='3d')
Z_real = y_test.reshape(X_test_grid.shape)
surf1 = ax5.plot_surface(X_test_grid, Y_test_grid, Z_real,
                         cmap='viridis', alpha=0.8, edgecolor='none')
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel('y', fontsize=10)
ax5.set_zlabel('f(x,y)', fontsize=10)
ax5.set_title('Função Real\nf(x,y) = sin(x)*cos(y) + 0.1*x*y',
              fontsize=12, fontweight='bold')
fig.colorbar(surf1, ax=ax5, shrink=0.5)

# 5.6 Superfície predita
ax6 = plt.subplot(2, 4, 6, projection='3d')
Z_pred = y_pred_test.reshape(X_test_grid.shape)
surf2 = ax6.plot_surface(X_test_grid, Y_test_grid, Z_pred,
                         cmap='viridis', alpha=0.8, edgecolor='none')
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('y', fontsize=10)
ax6.set_zlabel('f(x,y)', fontsize=10)
ax6.set_title('Função Aproximada (ANFIS)', fontsize=12, fontweight='bold')
fig.colorbar(surf2, ax=ax6, shrink=0.5)

# 5.7 Erro absoluto
ax7 = plt.subplot(2, 4, 7, projection='3d')
Z_erro = np.abs(Z_real - Z_pred)
surf3 = ax7.plot_surface(X_test_grid, Y_test_grid, Z_erro,
                         cmap='Reds', alpha=0.8, edgecolor='none')
ax7.set_xlabel('x', fontsize=10)
ax7.set_ylabel('y', fontsize=10)
ax7.set_zlabel('Erro', fontsize=10)
ax7.set_title(f'Erro Absoluto\nMáx={Z_erro.max():.4f}',
              fontsize=12, fontweight='bold')
fig.colorbar(surf3, ax=ax7, shrink=0.5)

# 5.8 Distribuição de erros
ax8 = plt.subplot(2, 4, 8)
erros = y_test - y_pred_test
ax8.hist(erros, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax8.set_xlabel('Erro (Real - Predito)', fontsize=11)
ax8.set_ylabel('Frequência', fontsize=11)
ax8.set_title(f'Distribuição de Erros\nMédia={erros.mean():.6f}, Std={erros.std():.6f}',
              fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/anfis_aproximacao.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráficos salvos em /tmp/anfis_aproximacao.png")

# ============================================================================
# 6. SALVAR MODELO
# ============================================================================
print("\n6. Salvando modelo...")

anfis.save('/tmp/anfis_modelo.npz')
print("   ✓ Modelo salvo em /tmp/anfis_modelo.npz")

# Testar carregamento
anfis_loaded = ANFIS.load('/tmp/anfis_modelo.npz')
y_loaded = anfis_loaded.predict(X_test[:5])
print(f"   ✓ Modelo carregado e testado")
print(f"   - Predição idêntica: {np.allclose(y_pred_test[:5], y_loaded)}")

# ============================================================================
# 7. RESUMO
# ============================================================================
print("\n" + "="*70)
print("RESUMO DO TREINAMENTO")
print("="*70)
print(f"""
✓ Problema: Aproximação de função não-linear 2D
✓ Modelo: ANFIS com {anfis.n_rules} regras
✓ Treinamento: {len(history['train_loss'])} épocas

Performance:
• RMSE (treino): {rmse_train:.6f}
• RMSE (teste):  {rmse_test:.6f}
• MAE (teste):   {mae_test:.6f}
• R² (teste):    {r2_test:.6f}

Características:
• Estabilidade de Lyapunov: ✓
• Taxa de aprendizado adaptativa: ✓
• Regularização L2: ✓
• Early stopping: ✓

Modelo capaz de aproximar funções complexas com alta precisão!
""")

print("="*70)
plt.show()
