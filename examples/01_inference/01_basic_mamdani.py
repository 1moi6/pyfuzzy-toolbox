"""
Exemplo 01: Sistema Mamdani Básico

Este exemplo demonstra como criar um sistema Mamdani simples
para controle de temperatura.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fuzzy_systems as fis
import numpy as np

def main():
    print("=" * 60)
    print("Exemplo 1: Sistema Mamdani para Controle de Temperatura")
    print("=" * 60)

    # Criar sistema usando a função auxiliar
    system = fis.create_mamdani_system(
        input_specs={
            'temperatura': ((0, 40), {
                'fria': ('triangular', (0, 0, 20)),
                'morna': ('triangular', (10, 20, 30)),
                'quente': ('triangular', (20, 40, 40))
            })
        },
        output_specs={
            'ventilador': ((0, 100), {
                'lento': ('triangular', (0, 0, 50)),
                'medio': ('triangular', (25, 50, 75)),
                'rapido': ('triangular', (50, 100, 100))
            })
        },
        rules=[
            ({'temperatura': 'fria'}, {'ventilador': 'lento'}, 'AND'),
            ({'temperatura': 'morna'}, {'ventilador': 'medio'}, 'AND'),
            ({'temperatura': 'quente'}, {'ventilador': 'rapido'}, 'AND'),
        ],
        name="Controle de Temperatura"
    )

    print(f"\nSistema criado: {system}")
    print(f"Variáveis de entrada: {list(system.input_variables.keys())}")
    print(f"Variáveis de saída: {list(system.output_variables.keys())}")
    print(f"Número de regras: {len(system.rule_base)}\n")

    # Testa várias temperaturas
    test_temps = [5, 10, 15, 20, 25, 30, 35]

    print("Testando diferentes temperaturas:")
    print("-" * 60)
    print(f"{'Temperatura (°C)':<20} {'Velocidade Ventilador (%)':<30}")
    print("-" * 60)

    for temp in test_temps:
        output = system.evaluate({'temperatura': temp})
        speed = output['ventilador']
        print(f"{temp:^20.1f} {speed:^30.2f}")

    # Análise detalhada para uma temperatura
    print("\n" + "=" * 60)
    print("Análise Detalhada para Temperatura = 22°C")
    print("=" * 60)

    temp_test = 22
    detailed = system.evaluate_detailed({'temperatura': temp_test})

    print(f"\nEntradas fuzzificadas:")
    for var, terms in detailed['fuzzified_inputs'].items():
        print(f"  {var}:")
        for term, degree in terms.items():
            print(f"    {term}: {degree:.3f}")

    print(f"\nRegras ativadas:")
    for rule_info in detailed['activated_rules']:
        print(f"  Regra {rule_info['rule_index']}: {rule_info['rule']}")
        print(f"    Força de ativação: {rule_info['firing_strength']:.3f}")

    print(f"\nSaída final:")
    for var, value in detailed['outputs'].items():
        print(f"  {var}: {value:.2f}%")


if __name__ == "__main__":
    main()
