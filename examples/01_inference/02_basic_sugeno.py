"""
Exemplo 02: Sistema Sugeno Básico

Este exemplo demonstra como criar um sistema Sugeno (TSK)
para predição simples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fuzzy_systems as fis
import numpy as np

def main():
    print("=" * 60)
    print("Exemplo 2: Sistema Sugeno para Controle de Potência")
    print("=" * 60)

    # Criar sistema Sugeno de ordem 0 (saídas constantes)
    system = fis.SugenoSystem(name="Controle Sugeno", order=0)

    # Define variável de entrada: temperatura
    temp_var = fis.LinguisticVariable('temperatura', (0, 40))
    temp_var.add_term(fis.FuzzySet('baixa', 'triangular', (0, 0, 20)))
    temp_var.add_term(fis.FuzzySet('media', 'triangular', (10, 20, 30)))
    temp_var.add_term(fis.FuzzySet('alta', 'triangular', (20, 40, 40)))
    system.add_input(temp_var)

    # Define variável de saída (dummy para Sugeno)
    output_var = fis.LinguisticVariable('potencia', (0, 100))
    system.add_output(output_var)

    # Adiciona regras com consequentes constantes (ordem 0)
    system.add_rule(fis.FuzzyRule(
        {'temperatura': 'baixa'},
        {'potencia': 20.0},  # Saída constante
        'AND'
    ))

    system.add_rule(fis.FuzzyRule(
        {'temperatura': 'media'},
        {'potencia': 50.0},  # Saída constante
        'AND'
    ))

    system.add_rule(fis.FuzzyRule(
        {'temperatura': 'alta'},
        {'potencia': 80.0},  # Saída constante
        'AND'
    ))

    print(f"\nSistema criado: {system}")
    print(f"Ordem do sistema: {system.order} (0 = constantes)")
    print(f"Número de regras: {len(system.rule_base)}\n")

    # Testa várias temperaturas
    test_temps = np.linspace(0, 40, 11)

    print("Testando diferentes temperaturas:")
    print("-" * 60)
    print(f"{'Temperatura (°C)':<20} {'Potência (%)':<30}")
    print("-" * 60)

    for temp in test_temps:
        output = system.evaluate({'temperatura': temp})
        power = output['potencia']
        print(f"{temp:^20.1f} {power:^30.2f}")

    # Análise detalhada
    print("\n" + "=" * 60)
    print("Análise Detalhada para Temperatura = 18°C")
    print("=" * 60)

    temp_test = 18
    detailed = system.evaluate_detailed({'temperatura': temp_test})

    print(f"\nEntradas fuzzificadas:")
    for var, terms in detailed['fuzzified_inputs'].items():
        print(f"  {var}:")
        for term, degree in terms.items():
            print(f"    {term}: {degree:.3f}")

    print(f"\nRegras ativadas e suas contribuições:")
    total_weight = 0
    weighted_sum = 0

    for rule_info in detailed['activated_rules']:
        if rule_info['firing_strength'] > 0:
            print(f"  Regra {rule_info['rule_index']}:")
            print(f"    Força de ativação (wi): {rule_info['firing_strength']:.3f}")
            print(f"    Saída da regra (zi): {rule_info['rule_output']:.2f}")
            print(f"    Contribuição (wi × zi): {rule_info['weighted_output']:.2f}")

            total_weight += rule_info['firing_strength']
            weighted_sum += rule_info['weighted_output']

    print(f"\nCálculo da média ponderada:")
    print(f"  Soma ponderada: {weighted_sum:.3f}")
    print(f"  Soma dos pesos: {total_weight:.3f}")
    print(f"  Saída final: {weighted_sum / total_weight:.2f}%")

    print(f"\nSaída final (via compute):")
    for var, value in detailed['outputs'].items():
        print(f"  {var}: {value:.2f}%")


def example_sugeno_order_1():
    """
    Exemplo adicional: Sugeno de ordem 1 (funções lineares)
    """
    print("\n" + "=" * 60)
    print("Exemplo Extra: Sistema Sugeno de Ordem 1")
    print("=" * 60)

    # Sistema de ordem 1
    system = fis.SugenoSystem(name="Sugeno Ordem 1", order=1)

    # Entrada
    x_var = fis.LinguisticVariable('x', (0, 10))
    x_var.add_term(fis.FuzzySet('pequeno', 'triangular', (0, 0, 5)))
    x_var.add_term(fis.FuzzySet('grande', 'triangular', (5, 10, 10)))
    system.add_input(x_var)

    # Saída
    y_var = fis.LinguisticVariable('y', (0, 100))
    system.add_output(y_var)

    # Regras com funções lineares
    # IF x IS pequeno THEN y = 5 + 2*x
    system.add_rule(fis.FuzzyRule(
        {'x': 'pequeno'},
        {'y': {'const': 5, 'x': 2}},  # Função linear
        'AND'
    ))

    # IF x IS grande THEN y = 10 + 5*x
    system.add_rule(fis.FuzzyRule(
        {'x': 'grande'},
        {'y': {'const': 10, 'x': 5}},  # Função linear
        'AND'
    ))

    print(f"\nSistema de ordem 1 criado")
    print("Regra 1: IF x IS pequeno THEN y = 5 + 2*x")
    print("Regra 2: IF x IS grande THEN y = 10 + 5*x\n")

    # Testa
    test_values = [2, 4, 6, 8]
    print("Resultados:")
    print("-" * 40)
    print(f"{'x':<10} {'y':<10}")
    print("-" * 40)

    for x_val in test_values:
        output = system.compute({'x': x_val})
        print(f"{x_val:<10.1f} {output['y']:<10.2f}")


if __name__ == "__main__":
    main()
    example_sugeno_order_1()
