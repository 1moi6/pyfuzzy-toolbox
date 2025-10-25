"""
Exemplo Completo: Sistema Híbrido (Conhecimento + Dados)
==========================================================

Demonstra a integração completa entre módulos:
1. Criar FIS com conhecimento especialista (MamdaniSystem)
2. Converter para modelo otimizável (MamdaniLearning)
3. Ajustar parâmetros com dados observados
4. Exportar FIS otimizado
5. Comparar desempenho antes/depois

Problema: Controle de irrigação agrícola baseado em:
- Umidade do solo
- Temperatura ambiente
→ Tempo de irrigação (minutos)
"""

import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning.mamdani import MamdaniLearning
from fuzzy_systems.inference.rules import FuzzyRule

np.random.seed(42)

print("="*70)
print("SISTEMA HÍBRIDO: CONHECIMENTO ESPECIALISTA + DADOS")
print("="*70)

# ============================================================================
# 1. CRIAR FIS COM CONHECIMENTO ESPECIALISTA
# ============================================================================
print("\n1. Criando FIS inicial (conhecimento de agrônomo)...")

fis_inicial = MamdaniSystem(name="Controle de Irrigação v1.0")

# Umidade do solo (%)
umidade_solo = fis_inicial.add_input('umidade_solo', (0, 100))
umidade_solo.add_term('muito_seca', 'gaussian', (10, 8))
umidade_solo.add_term('seca', 'gaussian', (30, 8))
umidade_solo.add_term('adequada', 'gaussian', (50, 8))
umidade_solo.add_term('umida', 'gaussian', (70, 8))
umidade_solo.add_term('encharcada', 'gaussian', (90, 8))

# Temperatura (°C)
temperatura = fis_inicial.add_input('temperatura', (10, 40))
temperatura.add_term('fria', 'gaussian', (15, 4))
temperatura.add_term('amena', 'gaussian', (22, 4))
temperatura.add_term('quente', 'gaussian', (30, 4))
temperatura.add_term('muito_quente', 'gaussian', (37, 4))

# Tempo de irrigação (minutos)
irrigacao = fis_inicial.add_output('tempo_irrigacao', (0, 60))
irrigacao.add_term('nenhum', 'singleton', (0,))
irrigacao.add_term('curto', 'singleton', (10,))
irrigacao.add_term('medio', 'singleton', (25,))
irrigacao.add_term('longo', 'singleton', (45,))
irrigacao.add_term('muito_longo', 'singleton', (60,))

# Regras baseadas em conhecimento especialista
regras_especialista = [
    # Solo muito seco precisa muita água
    ({'umidade_solo': 'muito_seca', 'temperatura': 'fria'}, {'tempo_irrigacao': 'longo'}),
    ({'umidade_solo': 'muito_seca', 'temperatura': 'amena'}, {'tempo_irrigacao': 'muito_longo'}),
    ({'umidade_solo': 'muito_seca', 'temperatura': 'quente'}, {'tempo_irrigacao': 'muito_longo'}),

    # Solo seco
    ({'umidade_solo': 'seca', 'temperatura': 'fria'}, {'tempo_irrigacao': 'medio'}),
    ({'umidade_solo': 'seca', 'temperatura': 'amena'}, {'tempo_irrigacao': 'longo'}),
    ({'umidade_solo': 'seca', 'temperatura': 'quente'}, {'tempo_irrigacao': 'longo'}),

    # Solo adequado
    ({'umidade_solo': 'adequada', 'temperatura': 'fria'}, {'tempo_irrigacao': 'nenhum'}),
    ({'umidade_solo': 'adequada', 'temperatura': 'amena'}, {'tempo_irrigacao': 'curto'}),
    ({'umidade_solo': 'adequada', 'temperatura': 'quente'}, {'tempo_irrigacao': 'medio'}),

    # Solo úmido
    ({'umidade_solo': 'umida', 'temperatura': 'fria'}, {'tempo_irrigacao': 'nenhum'}),
    ({'umidade_solo': 'umida', 'temperatura': 'amena'}, {'tempo_irrigacao': 'nenhum'}),
    ({'umidade_solo': 'umida', 'temperatura': 'quente'}, {'tempo_irrigacao': 'curto'}),

    # Solo encharcado
    ({'umidade_solo': 'encharcada', 'temperatura': 'fria'}, {'tempo_irrigacao': 'nenhum'}),
    ({'umidade_solo': 'encharcada', 'temperatura': 'amena'}, {'tempo_irrigacao': 'nenhum'}),
    ({'umidade_solo': 'encharcada', 'temperatura': 'quente'}, {'tempo_irrigacao': 'nenhum'}),
]

for antecedente, consequente in regras_especialista:
    fis_inicial.rule_base.add_rule(FuzzyRule(antecedente, consequente))

print(f"   ✓ FIS criado com {len(fis_inicial.rule_base.rules)} regras")
print(f"   ✓ Baseado em conhecimento de agrônomo")

# ============================================================================
# 2. COLETAR DADOS REAIS DE CAMPO
# ============================================================================
print("\n2. Coletando dados reais de campo (sensores + resultados)...")

# Simular dados observados em campo (umidade, temp, tempo_ótimo)
# Dados mostram que o conhecimento especialista não está perfeitamente calibrado
n_samples = 100
X_observado = np.random.uniform([0, 10], [100, 40], (n_samples, 2))

# Gerar tempo "ótimo" baseado em função + ruído
# (simulando observação de resultados reais)
def tempo_otimo_real(umid, temp):
    """Função que simula o comportamento real ideal"""
    # Solo seco precisa mais água
    fator_umidade = np.clip(100 - umid, 0, 100) / 100
    # Temperatura alta precisa mais água
    fator_temp = np.clip(temp - 10, 0, 30) / 30
    # Combinação
    tempo = 60 * (0.6 * fator_umidade + 0.4 * fator_temp)
    return tempo

y_observado = tempo_otimo_real(X_observado[:, 0], X_observado[:, 1])
y_observado += np.random.normal(0, 2, n_samples)  # Ruído de medição
y_observado = np.clip(y_observado, 0, 60)

print(f"   ✓ {n_samples} observações coletadas")
print(f"   ✓ Umidade: {X_observado[:, 0].min():.1f}-{X_observado[:, 0].max():.1f}%")
print(f"   ✓ Temperatura: {X_observado[:, 1].min():.1f}-{X_observado[:, 1].max():.1f}°C")
print(f"   ✓ Tempo irrigação: {y_observado.min():.1f}-{y_observado.max():.1f} min")

# ============================================================================
# 3. AVALIAR FIS INICIAL
# ============================================================================
print("\n3. Avaliando FIS inicial com conhecimento especialista...")

y_pred_inicial = []
for x in X_observado:
    resultado = fis_inicial.evaluate(umidade_solo=x[0], temperatura=x[1])
    y_pred_inicial.append(resultado['tempo_irrigacao'])
y_pred_inicial = np.array(y_pred_inicial)

rmse_inicial = np.sqrt(np.mean((y_observado - y_pred_inicial)**2))
mae_inicial = np.mean(np.abs(y_observado - y_pred_inicial))

print(f"   - RMSE: {rmse_inicial:.2f} minutos")
print(f"   - MAE:  {mae_inicial:.2f} minutos")
print(f"   ✓ Sistema funciona, mas pode melhorar")

# ============================================================================
# 4. CONVERTER FIS → MamdaniLearning
# ============================================================================
print("\n4. Convertendo FIS para modelo otimizável...")

mamdani_learning = MamdaniLearning.from_mamdani_system(fis_inicial)

print(f"   ✓ Modelo convertido")
print(f"   - {mamdani_learning.n_inputs} entradas")
print(f"   - {mamdani_learning.n_rules} regras")
print(f"   - {mamdani_learning.n_mfs_output} termos de saída")

# ============================================================================
# 5. OTIMIZAR COM DADOS OBSERVADOS
# ============================================================================
print("\n5. Otimizando parâmetros com dados de campo...")
print("   (Ajustando MFs gaussianas e centroides...)")

# Treinar com gradiente
mamdani_learning.fit(
    X_observado, y_observado,
    epochs=100,
    learning_rate=0.05,
    batch_size=20,
    verbose=False
)

print("   ✓ Otimização por gradiente concluída")

# Otimização metaheurística adicional (consequentes)
print("   (Otimizando mapeamento de regras com PSO...)")

mamdani_learning.fit_metaheuristic(
    X_observado, y_observado,
    optimizer='pso',
    optimize_params='consequents_only',
    n_particles=30,
    n_iterations=50,
    verbose=False
)

print("   ✓ Otimização metaheurística concluída")

# ============================================================================
# 6. EXPORTAR FIS OTIMIZADO
# ============================================================================
print("\n6. Exportando FIS otimizado...")

fis_otimizado = mamdani_learning.to_mamdani_system(
    input_names=['umidade_solo', 'temperatura'],
    output_name='tempo_irrigacao'
)

print(f"   ✓ FIS otimizado exportado")
print(f"   ✓ Nome: {fis_otimizado.name}")

# ============================================================================
# 7. AVALIAR FIS OTIMIZADO
# ============================================================================
print("\n7. Avaliando FIS otimizado...")

y_pred_otimizado = []
for x in X_observado:
    resultado = fis_otimizado.evaluate(umidade_solo=x[0], temperatura=x[1])
    y_pred_otimizado.append(resultado['tempo_irrigacao'])
y_pred_otimizado = np.array(y_pred_otimizado)

rmse_otimizado = np.sqrt(np.mean((y_observado - y_pred_otimizado)**2))
mae_otimizado = np.mean(np.abs(y_observado - y_pred_otimizado))

print(f"   - RMSE: {rmse_otimizado:.2f} minutos")
print(f"   - MAE:  {mae_otimizado:.2f} minutos")

melhoria_rmse = (rmse_inicial - rmse_otimizado) / rmse_inicial * 100
melhoria_mae = (mae_inicial - mae_otimizado) / mae_inicial * 100

print(f"\n   Melhoria:")
print(f"   - RMSE: {melhoria_rmse:.1f}% redução")
print(f"   - MAE:  {melhoria_mae:.1f}% redução")

# ============================================================================
# 8. VISUALIZAÇÃO COMPARATIVA
# ============================================================================
print("\n8. Criando visualizações comparativas...")

fig = plt.figure(figsize=(16, 10))

# 8.1 Comparação de predições
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_observado, y_pred_inicial, alpha=0.5, label='FIS Inicial', s=30)
ax1.scatter(y_observado, y_pred_otimizado, alpha=0.5, label='FIS Otimizado', s=30)
lim = [0, 60]
ax1.plot(lim, lim, 'r--', linewidth=2, label='Ideal')
ax1.set_xlabel('Tempo Observado (min)', fontsize=11)
ax1.set_ylabel('Tempo Predito (min)', fontsize=11)
ax1.set_title('Predições vs Observações', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 8.2 Distribuição de erros
ax2 = plt.subplot(2, 3, 2)
erros_inicial = y_observado - y_pred_inicial
erros_otimizado = y_observado - y_pred_otimizado
ax2.hist(erros_inicial, bins=20, alpha=0.5, label='Inicial', edgecolor='black')
ax2.hist(erros_otimizado, bins=20, alpha=0.5, label='Otimizado', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Erro (Observado - Predito)', fontsize=11)
ax2.set_ylabel('Frequência', fontsize=11)
ax2.set_title('Distribuição de Erros', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 8.3 Métricas
ax3 = plt.subplot(2, 3, 3)
metricas = ['RMSE\n(min)', 'MAE\n(min)']
inicial_vals = [rmse_inicial, mae_inicial]
otimizado_vals = [rmse_otimizado, mae_otimizado]
x_pos = np.arange(len(metricas))
width = 0.35
ax3.bar(x_pos - width/2, inicial_vals, width, label='Inicial', alpha=0.7)
ax3.bar(x_pos + width/2, otimizado_vals, width, label='Otimizado', alpha=0.7)
ax3.set_ylabel('Valor', fontsize=11)
ax3.set_title('Comparação de Métricas', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metricas)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 8.4 Superfície FIS Inicial
ax4 = plt.subplot(2, 3, 4, projection='3d')
umid_range = np.linspace(0, 100, 30)
temp_range = np.linspace(10, 40, 30)
U, T = np.meshgrid(umid_range, temp_range)
Z_inicial = np.zeros_like(U)
for i in range(len(umid_range)):
    for j in range(len(temp_range)):
        res = fis_inicial.evaluate(umidade_solo=U[j, i], temperatura=T[j, i])
        Z_inicial[j, i] = res['tempo_irrigacao']
surf1 = ax4.plot_surface(U, T, Z_inicial, cmap='Blues', alpha=0.8)
ax4.set_xlabel('Umidade (%)', fontsize=10)
ax4.set_ylabel('Temp (°C)', fontsize=10)
ax4.set_zlabel('Tempo (min)', fontsize=10)
ax4.set_title('FIS Inicial\n(Conhecimento)', fontsize=12, fontweight='bold')

# 8.5 Superfície FIS Otimizado
ax5 = plt.subplot(2, 3, 5, projection='3d')
Z_otimizado = np.zeros_like(U)
for i in range(len(umid_range)):
    for j in range(len(temp_range)):
        res = fis_otimizado.evaluate(umidade_solo=U[j, i], temperatura=T[j, i])
        Z_otimizado[j, i] = res['tempo_irrigacao']
surf2 = ax5.plot_surface(U, T, Z_otimizado, cmap='Greens', alpha=0.8)
ax5.set_xlabel('Umidade (%)', fontsize=10)
ax5.set_ylabel('Temp (°C)', fontsize=10)
ax5.set_zlabel('Tempo (min)', fontsize=10)
ax5.set_title('FIS Otimizado\n(Conhecimento + Dados)', fontsize=12, fontweight='bold')

# 8.6 Diferença entre superfícies
ax6 = plt.subplot(2, 3, 6, projection='3d')
Z_diff = np.abs(Z_otimizado - Z_inicial)
surf3 = ax6.plot_surface(U, T, Z_diff, cmap='Reds', alpha=0.8)
ax6.set_xlabel('Umidade (%)', fontsize=10)
ax6.set_ylabel('Temp (°C)', fontsize=10)
ax6.set_zlabel('|Diferença|', fontsize=10)
ax6.set_title(f'Mudanças no FIS\nMáx={Z_diff.max():.1f} min', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/hibrido_conhecimento_dados.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráficos salvos em /tmp/hibrido_conhecimento_dados.png")

# ============================================================================
# 9. RESUMO
# ============================================================================
print("\n" + "="*70)
print("RESUMO DO PROCESSO HÍBRIDO")
print("="*70)
print(f"""
Fluxo de trabalho:
1. ✓ FIS inicial criado com conhecimento especialista
2. ✓ Dados reais coletados em campo ({n_samples} observações)
3. ✓ FIS convertido para modelo otimizável
4. ✓ Parâmetros ajustados com dados (gradiente + PSO)
5. ✓ FIS otimizado exportado

Performance:
                    Inicial    Otimizado    Melhoria
  RMSE (min):      {rmse_inicial:7.2f}    {rmse_otimizado:9.2f}    {melhoria_rmse:6.1f}%
  MAE (min):       {mae_inicial:7.2f}    {mae_otimizado:9.2f}    {melhoria_mae:6.1f}%

Vantagens da abordagem híbrida:
• Mantém conhecimento especialista como ponto de partida
• Ajusta automaticamente com dados reais
• Combina interpretabilidade (regras) com precisão (dados)
• FIS otimizado pronto para produção

Caso de uso ideal para:
✓ Sistemas onde há conhecimento prévio parcial
✓ Dados de campo disponíveis para validação
✓ Necessidade de explicabilidade (regras fuzzy)
""")

print("="*70)
plt.show()
