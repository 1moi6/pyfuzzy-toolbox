"""
Exemplo 11: Visualização de Sistemas Fuzzy

Este exemplo demonstra as funcionalidades de visualização:
- Plot de variáveis linguísticas individuais
- Plot de todas variáveis de um sistema
- Plot de resposta do sistema (2D)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fuzzy_systems as fs


def exemplo_1_variavel_simples():
    """
    Exemplo 1: Plotar uma variável linguística simples
    """
    print("=" * 70)
    print("EXEMPLO 1: Variável Linguística Simples")
    print("=" * 70)

    # Cria variável
    temp = fs.LinguisticVariable('Temperatura (°C)', (0, 40))
    temp.add_term('Fria', 'triangular', (0, 0, 20))
    temp.add_term('Morna', 'triangular', (10, 20, 30))
    temp.add_term('Quente', 'triangular', (20, 40, 40))

    print(f"\nVariável criada: {temp}")
    print("\nPlotando variável 'temperatura'...")

    # Plot simples
    temp.plot()


def exemplo_2_variavel_customizada():
    """
    Exemplo 2: Plot customizado com cores e estilos
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Variável com Plot Customizado")
    print("=" * 70)

    # Cria variável com mais termos
    umidade = fs.LinguisticVariable('Umidade (%)', (0, 100))
    umidade.add_term('Seca', 'trapezoidal', (0, 0, 30, 50))
    umidade.add_term('Normal', 'triangular', (40, 60, 80))
    umidade.add_term('Úmida', 'trapezoidal', (70, 85, 100, 100))

    print(f"\nVariável criada: {umidade}")
    print("\nPlotando com cores customizadas...")

    # Plot com customizações
    umidade.plot(
        colors=['orange', 'green', 'blue'],
        linewidth=3,
        alpha=0.8,
        figsize=(12, 6),
        title='Funções de Pertinência da Umidade'
    )


def exemplo_3_sistema_completo():
    """
    Exemplo 3: Sistema completo com múltiplas variáveis
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Sistema de Ar Condicionado")
    print("=" * 70)

    # Cria sistema
    system = fs.MamdaniSystem(name="Controle de AC")

    # Variáveis de entrada
    system.add_input('temperatura', (15, 35))
    system.add_input('umidade', (20, 90))

    # Variável de saída
    system.add_output('ventilador', (0, 100))

    # Termos de temperatura
    system.add_term('temperatura', 'fria', 'trapezoidal', (15, 15, 18, 22))
    system.add_term('temperatura', 'agradavel', 'triangular', (20, 24, 28))
    system.add_term('temperatura', 'quente', 'trapezoidal', (26, 30, 35, 35))

    # Termos de umidade
    system.add_term('umidade', 'seca', 'triangular', (20, 20, 50))
    system.add_term('umidade', 'normal', 'triangular', (40, 60, 80))
    system.add_term('umidade', 'umida', 'triangular', (70, 90, 90))

    # Termos de saída
    system.add_term('ventilador', 'desligado', 'triangular', (0, 0, 20))
    system.add_term('ventilador', 'baixo', 'triangular', (10, 30, 50))
    system.add_term('ventilador', 'medio', 'triangular', (40, 60, 80))
    system.add_term('ventilador', 'alto', 'triangular', (70, 100, 100))

    # Regras
    system.add_rules([
        (['fria', 'seca'], ['desligado']),
        (['fria', 'normal'], ['desligado']),
        (['fria', 'umida'], ['baixo']),
        (['agradavel', 'seca'], ['baixo']),
        (['agradavel', 'normal'], ['medio']),
        (['agradavel', 'umida'], ['medio']),
        (['quente', 'seca'], ['medio']),
        (['quente', 'normal'], ['alto']),
        (['quente', 'umida'], ['alto']),
    ])

    print(f"\nSistema criado: {system}")
    print(f"  Entradas: {list(system.input_variables.keys())}")
    print(f"  Saídas: {list(system.output_variables.keys())}")
    print(f"  Regras: {len(system.rule_base.rules)}")

    # Plot de todas as variáveis
    print("\nPlotando todas as variáveis do sistema...")
    system.plot_variables()


def exemplo_4_resposta_sistema():
    """
    Exemplo 4: Resposta do sistema (gráfico 2D)
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Resposta do Sistema (2D)")
    print("=" * 70)

    # Sistema simples
    system = fs.MamdaniSystem(name="Controle Simples")

    system.add_input('temperatura', (0, 40))
    system.add_output('ventilador', (0, 100))

    system.add_term('temperatura', 'fria', 'triangular', (0, 0, 20))
    system.add_term('temperatura', 'morna', 'triangular', (10, 20, 30))
    system.add_term('temperatura', 'quente', 'triangular', (20, 40, 40))

    system.add_term('ventilador', 'lento', 'triangular', (0, 0, 50))
    system.add_term('ventilador', 'medio', 'triangular', (25, 50, 75))
    system.add_term('ventilador', 'rapido', 'triangular', (50, 100, 100))

    system.add_rules([
        (['fria'], ['lento']),
        (['morna'], ['medio']),
        (['quente'], ['rapido']),
    ])

    print(f"\nSistema criado: {system}")
    print("\nPlotando resposta do sistema (temperatura vs ventilador)...")

    # Plot da resposta
    system.plot_output('temperatura', 'ventilador', num_points=200)


def exemplo_5_apenas_entradas():
    """
    Exemplo 5: Plotar apenas variáveis de entrada
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Apenas Variáveis de Entrada")
    print("=" * 70)

    system = fs.MamdaniSystem()

    system.add_input('temp', (0, 40))
    system.add_input('umid', (0, 100))
    system.add_output('vent', (0, 100))

    system.add_term('temp', 'baixa', 'triangular', (0, 0, 20))
    system.add_term('temp', 'alta', 'triangular', (20, 40, 40))

    system.add_term('umid', 'seca', 'triangular', (0, 0, 50))
    system.add_term('umid', 'umida', 'triangular', (50, 100, 100))

    system.add_term('vent', 'lento', 'triangular', (0, 0, 50))
    system.add_term('vent', 'rapido', 'triangular', (50, 100, 100))

    print(f"\nSistema criado: {system}")
    print("\nPlotando apenas entradas...")

    # Plot apenas entradas
    system.plot_variables('input')


def exemplo_6_tipos_funcoes():
    """
    Exemplo 6: Diferentes tipos de funções de pertinência
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Tipos de Funções de Pertinência")
    print("=" * 70)

    velocidade = fs.LinguisticVariable('Velocidade (km/h)', (0, 120))

    # Diferentes tipos de funções
    velocidade.add_term('Muito Lenta', 'gaussian', (10, 8))
    velocidade.add_term('Lenta', 'triangular', (20, 40, 60))
    velocidade.add_term('Normal', 'generalized_bell', (15, 3, 70))
    velocidade.add_term('Rápida', 'triangular', (80, 90, 100))
    velocidade.add_term('Muito Rápida', 'sigmoid', (0.2, 100))

    print(f"\nVariável criada: {velocidade}")
    print("\nTipos de funções:")
    for term_name, fuzzy_set in velocidade.terms.items():
        print(f"  - {term_name}: {fuzzy_set.mf_type}")

    print("\nPlotando diferentes tipos de funções...")
    velocidade.plot(figsize=(14, 6))


def main():
    print("\n")
    print("*" * 70)
    print(" " * 20 + "VISUALIZAÇÃO DE SISTEMAS FUZZY")
    print("*" * 70)

    # Roda todos os exemplos
    exemplo_1_variavel_simples()

    input("\nPressione Enter para continuar para o Exemplo 2...")
    exemplo_2_variavel_customizada()

    input("\nPressione Enter para continuar para o Exemplo 3...")
    exemplo_3_sistema_completo()

    input("\nPressione Enter para continuar para o Exemplo 4...")
    exemplo_4_resposta_sistema()

    input("\nPressione Enter para continuar para o Exemplo 5...")
    exemplo_5_apenas_entradas()

    input("\nPressione Enter para continuar para o Exemplo 6...")
    exemplo_6_tipos_funcoes()

    print("\n" + "=" * 70)
    print("RESUMO DAS FUNCIONALIDADES")
    print("=" * 70)

    print("""
✅ TRÊS NÍVEIS DE VISUALIZAÇÃO IMPLEMENTADOS:

1. LinguisticVariable.plot()
   → Plota funções de pertinência de uma variável
   → Customizável (cores, tamanho, estilo)

2. System.plot_variables(var_type)
   → Plota todas variáveis do sistema
   → var_type: 'input', 'output', ou 'all'
   → Cria subplots automaticamente

3. System.plot_output(input_var, output_var)
   → Plota resposta do sistema (2D)
   → Para sistemas com múltiplas entradas, fixa as outras no ponto médio

💡 PRÓXIMOS PASSOS (Implementação Futura):
   • plot_control_surface() - Superfície 3D (2 entradas)
   • plot_inference() - Processo de inferência passo a passo
   • Animações interativas

🎨 CUSTOMIZAÇÃO:
   Todos os métodos aceitam **kwargs para customizar:
   - colors, linewidth, alpha
   - figsize, grid, title
   - E muito mais!
    """)

    print("\n" + "=" * 70)
    print("FIM DO EXEMPLO")
    print("=" * 70)


if __name__ == "__main__":
    main()
