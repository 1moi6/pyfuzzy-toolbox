"""
Teste da nova implementação simplificada de custom dynamics
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/1moi6/Desktop/Minicurso Fuzzy/fuzzy_systems')

import fuzzy_systems as fs

# Criar um FIS simples
fis = fs.inference.MamdaniSystem()

# Adicionar variáveis
x_var = fs.core.LinguisticVariable('x', (0, 100))
x_var.add_mf('low', 'trimf', [0, 0, 50])
x_var.add_mf('high', 'trimf', [50, 100, 100])
fis.add_input_variable(x_var)

dx_var = fs.core.LinguisticVariable('dx', (-10, 10))
dx_var.add_mf('negative', 'trimf', [-10, -10, 0])
dx_var.add_mf('positive', 'trimf', [0, 10, 10])
fis.add_output_variable(dx_var)

# Adicionar regras
fis.add_rule([('x', 'low')], [('dx', 'positive')])
fis.add_rule([('x', 'high')], [('dx', 'negative')])

print("=" * 60)
print("TESTE 1: Discrete System - Modo Custom")
print("=" * 60)

# Função customizada SIMPLES
def my_dynamics(states, fis_outputs):
    """
    Dinâmica customizada simples:
    x_{n+1} = x_n + 0.1 * x_n * f(x_n)
    """
    x = states[0]
    f = fis_outputs[0]
    return [x + 0.1 * x * f]

# Testar sistema discreto
pfuzzy_discrete = fs.dynamic.PFuzzyDiscrete(
    fis,
    mode='custom',
    dynamic_function=my_dynamics
)

print(f"✓ Sistema discreto criado com sucesso")
print(f"  - State vars: {pfuzzy_discrete.state_vars}")
print(f"  - Mode: {pfuzzy_discrete.mode}")

# Simular
n, traj = pfuzzy_discrete.simulate(x0={'x': 10.0}, n_steps=10)

print(f"✓ Simulação discreta concluída")
print(f"  - Steps: {len(n)}")
print(f"  - Trajetória inicial: {traj[0]}")
print(f"  - Trajetória final: {traj[-1]}")

print("\n" + "=" * 60)
print("TESTE 2: Continuous System - Modo Custom")
print("=" * 60)

# Função customizada para sistema contínuo
def my_continuous_dynamics(states, fis_outputs):
    """
    Dinâmica contínua customizada:
    dx/dt = 0.05 * x * f(x)
    """
    x = states[0]
    f = fis_outputs[0]
    return [0.05 * x * f]

# Testar sistema contínuo
pfuzzy_continuous = fs.dynamic.PFuzzyContinuous(
    fis,
    mode='custom',
    dynamic_function=my_continuous_dynamics
)

print(f"✓ Sistema contínuo criado com sucesso")
print(f"  - State vars: {pfuzzy_continuous.state_vars}")
print(f"  - Mode: {pfuzzy_continuous.mode}")

# Simular
t, traj = pfuzzy_continuous.simulate(
    x0={'x': 10.0},
    t_span=(0, 5),
    dt=0.1
)

print(f"✓ Simulação contínua concluída")
print(f"  - Time points: {len(t)}")
print(f"  - Trajetória inicial: {traj[0]}")
print(f"  - Trajetória final: {traj[-1]}")

print("\n" + "=" * 60)
print("TESTE 3: Performance - Comparar modos")
print("=" * 60)

import time

# Teste absolute mode
pfuzzy_abs = fs.dynamic.PFuzzyDiscrete(fis, mode='absolute')
t0 = time.time()
n, traj_abs = pfuzzy_abs.simulate(x0={'x': 10.0}, n_steps=1000)
t_abs = time.time() - t0

# Teste custom mode
t0 = time.time()
n, traj_custom = pfuzzy_discrete.simulate(x0={'x': 10.0}, n_steps=1000)
t_custom = time.time() - t0

print(f"✓ Absolute mode: {t_abs*1000:.2f} ms")
print(f"✓ Custom mode:   {t_custom*1000:.2f} ms")
print(f"✓ Overhead:      {(t_custom/t_abs - 1)*100:.1f}%")

print("\n" + "=" * 60)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 60)
