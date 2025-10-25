"""
Exemplo Completo: Sistema de Controle de Temperatura (Mamdani)
===============================================================

Demonstra um sistema fuzzy completo do zero:
1. Criação do FIS
2. Definição de variáveis linguísticas
3. Criação de regras
4. Inferência e defuzzificação
5. Visualização

Problema: Controlar a velocidade de um ventilador baseado em:
- Temperatura ambiente
- Umidade relativa
"""

import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.inference.rules import FuzzyRule

print("="*70)
print("CONTROLE FUZZY DE TEMPERATURA")
print("="*70)

# ============================================================================
# 1. CRIAR SISTEMA MAMDANI
# ============================================================================
print("\n1. Criando sistema de inferência Mamdani...")

fis = MamdaniSystem(name="Controle de Temperatura")

# ============================================================================
# 2. DEFINIR VARIÁVEIS DE ENTRADA
# ============================================================================
print("\n2. Definindo variáveis de entrada...")

# Temperatura (°C)
temperatura = fis.add_input('temperatura', (0, 40))
temperatura.add_term('fria', 'trapezoidal', (0, 0, 15, 20))
temperatura.add_term('agradavel', 'triangular', (18, 24, 30))
temperatura.add_term('quente', 'trapezoidal', (28, 35, 40, 40))

# Umidade (%)
umidade = fis.add_input('umidade', (0, 100))
umidade.add_term('seca', 'trapezoidal', (0, 0, 30, 50))
umidade.add_term('confortavel', 'triangular', (30, 50, 70))
umidade.add_term('umida', 'trapezoidal', (60, 80, 100, 100))

print(f"   ✓ Temperatura: {len(temperatura.terms)} termos")
print(f"   ✓ Umidade: {len(umidade.terms)} termos")

# ============================================================================
# 3. DEFINIR VARIÁVEL DE SAÍDA
# ============================================================================
print("\n3. Definindo variável de saída...")

# Velocidade do ventilador (%)
ventilador = fis.add_output('ventilador', (0, 100))
ventilador.add_term('desligado', 'triangular', (0, 0, 25))
ventilador.add_term('lento', 'triangular', (10, 30, 50))
ventilador.add_term('medio', 'triangular', (40, 60, 80))
ventilador.add_term('rapido', 'triangular', (70, 100, 100))

print(f"   ✓ Ventilador: {len(ventilador.terms)} termos")

# ============================================================================
# 4. DEFINIR REGRAS
# ============================================================================
print("\n4. Definindo regras...")

regras = [
    # Se está frio, ventilador desligado
    ({'temperatura': 'fria', 'umidade': 'seca'}, {'ventilador': 'desligado'}),
    ({'temperatura': 'fria', 'umidade': 'confortavel'}, {'ventilador': 'desligado'}),
    ({'temperatura': 'fria', 'umidade': 'umida'}, {'ventilador': 'lento'}),

    # Temperatura agradável
    ({'temperatura': 'agradavel', 'umidade': 'seca'}, {'ventilador': 'desligado'}),
    ({'temperatura': 'agradavel', 'umidade': 'confortavel'}, {'ventilador': 'lento'}),
    ({'temperatura': 'agradavel', 'umidade': 'umida'}, {'ventilador': 'medio'}),

    # Está quente, precisa ventilar
    ({'temperatura': 'quente', 'umidade': 'seca'}, {'ventilador': 'medio'}),
    ({'temperatura': 'quente', 'umidade': 'confortavel'}, {'ventilador': 'rapido'}),
    ({'temperatura': 'quente', 'umidade': 'umida'}, {'ventilador': 'rapido'}),
]

for antecedente, consequente in regras:
    fis.rule_base.add_rule(FuzzyRule(antecedente, consequente))

print(f"   ✓ {len(regras)} regras adicionadas")

# ============================================================================
# 5. TESTAR SISTEMA
# ============================================================================
print("\n5. Testando sistema...")
print("\n   Temperatura | Umidade | Ventilador")
print("   " + "-"*42)

# Casos de teste
casos_teste = [
    (15, 40, "Dia frio e seco"),
    (24, 50, "Confortável"),
    (28, 80, "Quente e úmido"),
    (35, 60, "Muito quente"),
    (18, 85, "Frio mas úmido"),
]

resultados = []
for temp, umid, desc in casos_teste:
    resultado = fis.evaluate(temperatura=temp, umidade=umid)
    vel_vent = resultado['ventilador']
    resultados.append((temp, umid, vel_vent))
    print(f"   {temp:5.0f}°C     | {umid:3.0f}%    | {vel_vent:5.1f}% - {desc}")

# ============================================================================
# 6. SUPERFÍCIE DE RESPOSTA
# ============================================================================
print("\n6. Gerando superfície de resposta...")

# Grid de entrada
temp_range = np.linspace(0, 40, 40)
umid_range = np.linspace(0, 100, 40)
T, U = np.meshgrid(temp_range, umid_range)

# Calcular saída para cada ponto
V = np.zeros_like(T)
for i in range(len(temp_range)):
    for j in range(len(umid_range)):
        resultado = fis.evaluate(temperatura=T[j, i], umidade=U[j, i])
        V[j, i] = resultado['ventilador']

# ============================================================================
# 7. VISUALIZAÇÃO
# ============================================================================
print("\n7. Criando visualizações...")

fig = plt.figure(figsize=(16, 10))

# 7.1 Funções de pertinência - Temperatura
ax1 = plt.subplot(2, 3, 1)
x_temp = np.linspace(0, 40, 200)
for termo, fuzzy_set in temperatura.terms.items():
    y = fuzzy_set.membership(x_temp)
    ax1.plot(x_temp, y, label=termo, linewidth=2)
ax1.set_xlabel('Temperatura (°C)', fontsize=11)
ax1.set_ylabel('Grau de Pertinência', fontsize=11)
ax1.set_title('Variável de Entrada: Temperatura', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 7.2 Funções de pertinência - Umidade
ax2 = plt.subplot(2, 3, 2)
x_umid = np.linspace(0, 100, 200)
for termo, fuzzy_set in umidade.terms.items():
    y = fuzzy_set.membership(x_umid)
    ax2.plot(x_umid, y, label=termo, linewidth=2)
ax2.set_xlabel('Umidade (%)', fontsize=11)
ax2.set_ylabel('Grau de Pertinência', fontsize=11)
ax2.set_title('Variável de Entrada: Umidade', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7.3 Funções de pertinência - Ventilador
ax3 = plt.subplot(2, 3, 3)
x_vent = np.linspace(0, 100, 200)
for termo, fuzzy_set in ventilador.terms.items():
    y = fuzzy_set.membership(x_vent)
    ax3.plot(x_vent, y, label=termo, linewidth=2)
ax3.set_xlabel('Velocidade (%)', fontsize=11)
ax3.set_ylabel('Grau de Pertinência', fontsize=11)
ax3.set_title('Variável de Saída: Ventilador', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 7.4 Superfície de resposta 3D
ax4 = plt.subplot(2, 3, 4, projection='3d')
surf = ax4.plot_surface(T, U, V, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax4.set_xlabel('Temperatura (°C)', fontsize=10)
ax4.set_ylabel('Umidade (%)', fontsize=10)
ax4.set_zlabel('Ventilador (%)', fontsize=10)
ax4.set_title('Superfície de Resposta 3D', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax4, shrink=0.5)

# 7.5 Superfície de resposta 2D (contorno)
ax5 = plt.subplot(2, 3, 5)
contour = ax5.contourf(T, U, V, levels=20, cmap='coolwarm')
ax5.contour(T, U, V, levels=10, colors='black', linewidths=0.5, alpha=0.3)

# Marcar casos de teste
for temp, umid, vel in resultados:
    ax5.plot(temp, umid, 'wo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax5.text(temp, umid+3, f'{vel:.0f}%', ha='center', fontsize=8, fontweight='bold')

ax5.set_xlabel('Temperatura (°C)', fontsize=11)
ax5.set_ylabel('Umidade (%)', fontsize=11)
ax5.set_title('Mapa de Contorno\n(pontos brancos = casos testados)', fontsize=12, fontweight='bold')
fig.colorbar(contour, ax=ax5, label='Ventilador (%)')

# 7.6 Resposta para diferentes umidades
ax6 = plt.subplot(2, 3, 6)
umidades_teste = [20, 50, 80]
cores = ['blue', 'green', 'red']

for umid_test, cor in zip(umidades_teste, cores):
    velocidades = []
    for temp in temp_range:
        resultado = fis.evaluate(temperatura=temp, umidade=umid_test)
        velocidades.append(resultado['ventilador'])
    ax6.plot(temp_range, velocidades, label=f'Umidade = {umid_test}%',
            color=cor, linewidth=2, marker='o', markersize=3)

ax6.set_xlabel('Temperatura (°C)', fontsize=11)
ax6.set_ylabel('Velocidade do Ventilador (%)', fontsize=11)
ax6.set_title('Resposta para Diferentes Umidades', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/controle_temperatura_mamdani.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráficos salvos em /tmp/controle_temperatura_mamdani.png")

# ============================================================================
# 8. RESUMO
# ============================================================================
print("\n" + "="*70)
print("RESUMO DO SISTEMA")
print("="*70)
print(f"""
✓ Sistema: {fis.name}
✓ Entradas: {len(fis.input_variables)} variáveis
  - temperatura: {len(temperatura.terms)} termos linguísticos
  - umidade: {len(umidade.terms)} termos linguísticos
✓ Saídas: {len(fis.output_variables)} variável
  - ventilador: {len(ventilador.terms)} termos linguísticos
✓ Regras: {len(fis.rule_base.rules)} regras

Casos de teste demonstram comportamento esperado:
• Frio → Ventilador desligado/lento
• Confortável → Ventilador lento/médio
• Quente → Ventilador médio/rápido
• Umidade alta → Aumenta velocidade

Sistema pronto para uso em aplicações reais!
""")

print("="*70)
plt.show()
