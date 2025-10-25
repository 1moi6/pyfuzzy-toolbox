"""
Exemplo Simples: Sistema p-Fuzzy
=================================

Demonstra o uso básico de sistemas p-fuzzy com código mínimo.
"""

import sys
sys.path.insert(0, '/Users/1moi6/Desktop/Minicurso Fuzzy/fuzzy_systems')

import numpy as np
import fuzzy_systems as fs
import matplotlib.pyplot as plt

print("=" * 70)
print("Exemplo Simples: Sistema p-Fuzzy")
print("=" * 70)

# ============================================================================
# Passo 1: Criar Sistema de Inferência Fuzzy
# ============================================================================

print("\n1. Criando FIS...")

fis = fs.MamdaniSystem(name="Modelo Simples")

# Entrada: estado atual
fis.add_input('x', (0, 10))
fis.add_term('x', 'baixo', 'triangular', (0, 0, 5))
fis.add_term('x', 'medio', 'triangular', (2, 5, 8))
fis.add_term('x', 'alto', 'triangular', (5, 10, 10))

# Saída: taxa de mudança
fis.add_output('dx', (-2, 2))
fis.add_term('dx', 'diminui', 'triangular', (-2, -2, 0))
fis.add_term('dx', 'estavel', 'triangular', (-0.5, 0, 0.5))
fis.add_term('dx', 'aumenta', 'triangular', (0, 2, 2))

# Regras simples:
# - Se x é baixo → aumenta
# - Se x é médio → estável
# - Se x é alto → diminui
fis.add_rule({'x': 'baixo'}, {'dx': 'aumenta'})
fis.add_rule({'x': 'medio'}, {'dx': 'estavel'})
fis.add_rule({'x': 'alto'}, {'dx': 'diminui'})

print(f"✅ FIS criado: {len(fis.rule_base.rules)} regras")


# ============================================================================
# Passo 2: Criar Sistema p-Fuzzy
# ============================================================================

print("\n2. Criando sistema p-fuzzy...")

# Modo contínuo: dx/dt = f(x)
# state_vars é inferido automaticamente das entradas do FIS!
pfuzzy = fs.dynamic.PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    method='rk4'
)

print("✅ Sistema p-fuzzy criado")


# ============================================================================
# Passo 3: Simular
# ============================================================================

print("\n3. Simulando...")

# x0 pode ser dicionário, lista, tupla ou array!
trajectory = pfuzzy.simulate(
    x0=[1],           # Condição inicial (também aceita {'x': 1}, (1,), etc.)
    t_span=(0, 20),   # Tempo: 0 a 20
    dt=0.01           # Passo de integração
)

print(f"✅ Simulação concluída!")
print(f"   x(0) = {trajectory[0, 0]:.2f}")
print(f"   x(20) = {trajectory[-1, 0]:.2f}")


# ============================================================================
# Passo 4: Visualizar
# ============================================================================

print("\n4. Gerando visualizações...")

# Plot 1: Trajetória
pfuzzy.plot_trajectory(title='Evolução Temporal')
plt.savefig('/tmp/pfuzzy_simple_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Trajetória salva em: /tmp/pfuzzy_simple_trajectory.png")

# Plot 2: Superfície de controle
x_vals = np.linspace(0, 10, 100)
dx_vals = [fis.evaluate({'x': x})['dx'] for x in x_vals]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, dx_vals, 'b-', linewidth=3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Equilíbrio (dx=0)')
ax.set_xlabel('Estado x', fontsize=12)
ax.set_ylabel('Taxa de mudança dx/dt', fontsize=12)
ax.set_title('Superfície de Controle Fuzzy', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/pfuzzy_simple_control.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Superfície salva em: /tmp/pfuzzy_simple_control.png")


# ============================================================================
# Resumo
# ============================================================================

print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

print("""
✅ Sistema p-fuzzy criado com sucesso!

📝 Estrutura:
   • 1 entrada (x) → estado do sistema
   • 1 saída (dx) → taxa de mudança
   • 3 regras → lógica de controle

🔄 Dinâmica:
   • Equação: dx/dt = f(x)
   • f(x) é determinado por regras fuzzy
   • Sistema busca equilíbrio em x ≈ 5

💡 Conceitos-chave:
   • As ENTRADAS do FIS são as variáveis de ESTADO
   • As SAÍDAS do FIS são as TAXAS DE MUDANÇA
   • Não precisa especificar mapeamento - é automático!
   • A ordem das saídas corresponde à ordem dos estados

🎯 Uso típico (SUPER SIMPLES!):

   # 1. Criar FIS com entradas = estados, saídas = taxas
   fis = fs.MamdaniSystem()
   fis.add_input('population', ...)    # Estado
   fis.add_output('growth', ...)       # Taxa de mudança

   # 2. Criar p-fuzzy (inferência automática!)
   pfuzzy = fs.dynamic.PFuzzyContinuous(fis)  # ← Só isso!

   # 3. Simular (x0 aceita dict, lista, tupla ou array)
   trajectory = pfuzzy.simulate(x0=[10], t_span=(0, 50))

   # 4. Visualizar
   pfuzzy.plot_trajectory()
""")

print("=" * 70)
print("✅ Exemplo concluído!")
print("=" * 70)
