"""
Exemplo 03: Problema Clássico da Gorjeta (Tipping Problem)

Sistema Mamdani para calcular gorjeta baseado na qualidade
do serviço e da comida.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fuzzy_systems as fis
import numpy as np

def main():
    print("=" * 60)
    print("Exemplo 3: Sistema de Gorjeta (Tipping Problem)")
    print("=" * 60)

    # Cria sistema Mamdani
    system = fis.MamdaniSystem(name="Sistema de Gorjeta")

    # Define variáveis de entrada

    # 1. Qualidade do serviço (0-10)
    servico = fis.LinguisticVariable('servico', (0, 10))
    servico.add_term(fis.FuzzySet('ruim', 'triangular', (0, 0, 5)))
    servico.add_term(fis.FuzzySet('bom', 'triangular', (0, 5, 10)))
    servico.add_term(fis.FuzzySet('excelente', 'triangular', (5, 10, 10)))
    system.add_input(servico)

    # 2. Qualidade da comida (0-10)
    comida = fis.LinguisticVariable('comida', (0, 10))
    comida.add_term(fis.FuzzySet('ruim', 'trapezoidal', (0, 0, 1, 3)))
    comida.add_term(fis.FuzzySet('boa', 'triangular', (2, 5, 8)))
    comida.add_term(fis.FuzzySet('excelente', 'trapezoidal', (7, 9, 10, 10)))
    system.add_input(comida)

    # Define variável de saída: gorjeta (0-25%)
    gorjeta = fis.LinguisticVariable('gorjeta', (0, 25))
    gorjeta.add_term(fis.FuzzySet('baixa', 'triangular', (0, 0, 13)))
    gorjeta.add_term(fis.FuzzySet('media', 'triangular', (0, 13, 25)))
    gorjeta.add_term(fis.FuzzySet('alta', 'triangular', (13, 25, 25)))
    system.add_output(gorjeta)

    # Define regras
    rules = [
        # Serviço ruim
        ({'servico': 'ruim', 'comida': 'ruim'}, {'gorjeta': 'baixa'}, 'AND'),
        ({'servico': 'ruim', 'comida': 'boa'}, {'gorjeta': 'baixa'}, 'AND'),
        ({'servico': 'ruim', 'comida': 'excelente'}, {'gorjeta': 'media'}, 'AND'),

        # Serviço bom
        ({'servico': 'bom', 'comida': 'ruim'}, {'gorjeta': 'baixa'}, 'AND'),
        ({'servico': 'bom', 'comida': 'boa'}, {'gorjeta': 'media'}, 'AND'),
        ({'servico': 'bom', 'comida': 'excelente'}, {'gorjeta': 'alta'}, 'AND'),

        # Serviço excelente
        ({'servico': 'excelente', 'comida': 'ruim'}, {'gorjeta': 'media'}, 'AND'),
        ({'servico': 'excelente', 'comida': 'boa'}, {'gorjeta': 'alta'}, 'AND'),
        ({'servico': 'excelente', 'comida': 'excelente'}, {'gorjeta': 'alta'}, 'AND'),
    ]

    for ant, cons, op in rules:
        system.add_rule(fis.FuzzyRule(ant, cons, op))

    print(f"\nSistema criado com {len(system.rule_base)} regras")

    # Testa cenários
    scenarios = [
        {'servico': 3, 'comida': 2, 'desc': 'Serviço ruim, comida ruim'},
        {'servico': 5, 'comida': 5, 'desc': 'Serviço médio, comida média'},
        {'servico': 7, 'comida': 8, 'desc': 'Serviço bom, comida excelente'},
        {'servico': 9, 'comida': 9, 'desc': 'Serviço excelente, comida excelente'},
        {'servico': 9, 'comida': 3, 'desc': 'Serviço excelente, comida ruim'},
    ]

    print("\n" + "=" * 80)
    print("Resultados para Diferentes Cenários")
    print("=" * 80)

    for scenario in scenarios:
        servico_val = scenario['servico']
        comida_val = scenario['comida']
        desc = scenario['desc']

        inputs = {'servico': servico_val, 'comida': comida_val}
        output = system.evaluate(inputs)
        tip = output['gorjeta']

        print(f"\n{desc}")
        print(f"  Serviço: {servico_val}/10, Comida: {comida_val}/10")
        print(f"  Gorjeta recomendada: {tip:.2f}%")

    # Análise detalhada de um caso
    print("\n" + "=" * 80)
    print("Análise Detalhada: Serviço = 6, Comida = 7")
    print("=" * 80)

    detailed = system.evaluate_detailed({'servico': 6, 'comida': 7})

    print("\nFuzzificação das entradas:")
    for var, terms in detailed['fuzzified_inputs'].items():
        print(f"\n  {var.capitalize()}:")
        for term, degree in terms.items():
            if degree > 0:
                print(f"    {term}: {degree:.3f}")

    print("\nRegras ativadas (com força > 0):")
    for rule_info in detailed['activated_rules']:
        print(f"\n  {rule_info['rule']}")
        print(f"    Força de ativação: {rule_info['firing_strength']:.3f}")

    print(f"\nGorjeta final: {detailed['outputs']['gorjeta']:.2f}%")

    # Análise de sensibilidade
    print("\n" + "=" * 80)
    print("Análise de Sensibilidade: Variando Serviço (Comida fixa em 7)")
    print("=" * 80)

    print(f"\n{'Serviço':<15} {'Gorjeta (%)':<15}")
    print("-" * 30)

    for serv in range(0, 11):
        output = system.evaluate({'servico': serv, 'comida': 7})
        print(f"{serv:<15} {output['gorjeta']:<15.2f}")


if __name__ == "__main__":
    main()
