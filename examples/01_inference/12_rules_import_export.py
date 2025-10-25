"""
Exemplo 12: Importação e Exportação de Regras

Este exemplo demonstra as funcionalidades de manipulação de regras:
- Conversão de regras para DataFrame Pandas
- Impressão de regras em diferentes estilos
- Exportação de regras para CSV, JSON, TXT
- Importação de regras de arquivos
- Estatísticas sobre regras
- Reutilização de regras entre diferentes sistemas
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fuzzy_systems as fs


def exemplo_1_visualizar_regras():
    """
    Exemplo 1: Diferentes formas de visualizar regras
    """
    print("=" * 70)
    print("EXEMPLO 1: Visualização de Regras")
    print("=" * 70)

    # Cria sistema simples
    system = fs.MamdaniSystem(name="Controle de Temperatura")

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

    print("\n1. ESTILO TABELA (padrão):")
    print("-" * 70)
    system.print_rules(style='table')

    print("\n2. ESTILO COMPACTO:")
    print("-" * 70)
    system.print_rules(style='compact', show_stats=False)

    print("\n3. ESTILO DETALHADO:")
    print("-" * 70)
    system.print_rules(style='detailed', show_stats=False)

    print("\n4. ESTILO NATURAL (IF-THEN):")
    print("-" * 70)
    system.print_rules(style='if-then', show_stats=False)


def exemplo_2_dataframe():
    """
    Exemplo 2: Conversão para DataFrame Pandas
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Regras como DataFrame Pandas")
    print("=" * 70)

    # Sistema de ar condicionado
    system = fs.MamdaniSystem(name="Controle de AC")

    system.add_input('temperatura', (15, 35))
    system.add_input('umidade', (20, 90))
    system.add_output('ventilador', (0, 100))

    system.add_term('temperatura', 'fria', 'trapezoidal', (15, 15, 18, 22))
    system.add_term('temperatura', 'agradavel', 'triangular', (20, 24, 28))
    system.add_term('temperatura', 'quente', 'trapezoidal', (26, 30, 35, 35))

    system.add_term('umidade', 'seca', 'triangular', (20, 20, 50))
    system.add_term('umidade', 'normal', 'triangular', (40, 60, 80))
    system.add_term('umidade', 'umida', 'triangular', (70, 90, 90))

    system.add_term('ventilador', 'desligado', 'triangular', (0, 0, 20))
    system.add_term('ventilador', 'baixo', 'triangular', (10, 30, 50))
    system.add_term('ventilador', 'medio', 'triangular', (40, 60, 80))
    system.add_term('ventilador', 'alto', 'triangular', (70, 100, 100))

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

    print("\nFormato COMPACTO (padrão):")
    print("-" * 70)
    df_compact = system.rules_to_dataframe(expand_variables=False)
    print(df_compact)

    print("\n\nFormato EXPANDIDO (uma coluna por variável):")
    print("-" * 70)
    df_expanded = system.rules_to_dataframe(expand_variables=True)
    print(df_expanded)

    print("\n\nESTATÍSTICAS DAS REGRAS:")
    print("-" * 70)
    stats = system.rules_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return system


def exemplo_3_exportar_regras(system):
    """
    Exemplo 3: Exportar regras para diferentes formatos
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Exportação de Regras")
    print("=" * 70)

    # Cria diretório temporário para exports
    export_dir = os.path.join(os.path.dirname(__file__), 'exported_rules')
    os.makedirs(export_dir, exist_ok=True)

    # Exporta para CSV
    csv_file = os.path.join(export_dir, 'regras_ac.csv')
    print(f"\n1. Exportando para CSV: {csv_file}")
    system.export_rules(csv_file, format='csv')
    print("   ✓ Exportado com sucesso!")

    # Exporta para JSON
    json_file = os.path.join(export_dir, 'regras_ac.json')
    print(f"\n2. Exportando para JSON: {json_file}")
    system.export_rules(json_file, format='json')
    print("   ✓ Exportado com sucesso!")

    # Exporta para TXT
    txt_file = os.path.join(export_dir, 'regras_ac.txt')
    print(f"\n3. Exportando para TXT: {txt_file}")
    system.export_rules(txt_file, format='txt')
    print("   ✓ Exportado com sucesso!")

    # Auto-detecção de formato pela extensão
    auto_file = os.path.join(export_dir, 'regras_ac_auto.csv')
    print(f"\n4. Exportando com auto-detecção: {auto_file}")
    system.export_rules(auto_file)  # format='auto' por padrão
    print("   ✓ Exportado com sucesso! (formato detectado: CSV)")

    print("\n" + "-" * 70)
    print("Arquivos criados:")
    for filename in os.listdir(export_dir):
        filepath = os.path.join(export_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  • {filename} ({size} bytes)")

    return export_dir


def exemplo_4_importar_regras(export_dir):
    """
    Exemplo 4: Importar regras de arquivos
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Importação de Regras")
    print("=" * 70)

    # Cria novo sistema vazio com as mesmas variáveis
    new_system = fs.MamdaniSystem(name="AC Importado")

    # Adiciona variáveis (necessário antes de importar regras)
    new_system.add_input('temperatura', (15, 35))
    new_system.add_input('umidade', (20, 90))
    new_system.add_output('ventilador', (0, 100))

    new_system.add_term('temperatura', 'fria', 'trapezoidal', (15, 15, 18, 22))
    new_system.add_term('temperatura', 'agradavel', 'triangular', (20, 24, 28))
    new_system.add_term('temperatura', 'quente', 'trapezoidal', (26, 30, 35, 35))

    new_system.add_term('umidade', 'seca', 'triangular', (20, 20, 50))
    new_system.add_term('umidade', 'normal', 'triangular', (40, 60, 80))
    new_system.add_term('umidade', 'umida', 'triangular', (70, 90, 90))

    new_system.add_term('ventilador', 'desligado', 'triangular', (0, 0, 20))
    new_system.add_term('ventilador', 'baixo', 'triangular', (10, 30, 50))
    new_system.add_term('ventilador', 'medio', 'triangular', (40, 60, 80))
    new_system.add_term('ventilador', 'alto', 'triangular', (70, 100, 100))

    print("\nSistema ANTES da importação:")
    print(f"  Número de regras: {len(new_system.rule_base.rules)}")

    # Importa de CSV
    csv_file = os.path.join(export_dir, 'regras_ac.csv')
    print(f"\nImportando regras de: {csv_file}")
    new_system.import_rules(csv_file)

    print(f"\nSistema DEPOIS da importação:")
    print(f"  Número de regras: {len(new_system.rule_base.rules)}")

    print("\nRegras importadas:")
    print("-" * 70)
    new_system.print_rules(style='compact', show_stats=False)

    return new_system


def exemplo_5_testar_sistema_importado(system_original, system_importado):
    """
    Exemplo 5: Verificar que sistema importado funciona identicamente
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Teste de Sistema Importado")
    print("=" * 70)

    # Casos de teste
    test_cases = [
        {'temperatura': 18, 'umidade': 30},
        {'temperatura': 24, 'umidade': 60},
        {'temperatura': 32, 'umidade': 85},
    ]

    print("\nComparando resultados dos dois sistemas:")
    print("-" * 70)
    print(f"{'Temp':<8} {'Umid':<8} {'Original':<12} {'Importado':<12} {'Status':<10}")
    print("-" * 70)

    all_match = True
    for inputs in test_cases:
        output_orig = system_original.compute(inputs)
        output_import = system_importado.compute(inputs)

        match = abs(output_orig['ventilador'] - output_import['ventilador']) < 0.01
        status = "✓ OK" if match else "✗ DIFERENTE"

        print(f"{inputs['temperatura']:<8} {inputs['umidade']:<8} "
              f"{output_orig['ventilador']:<12.2f} {output_import['ventilador']:<12.2f} "
              f"{status:<10}")

        all_match = all_match and match

    print("-" * 70)
    if all_match:
        print("✓ SUCESSO: Sistema importado funciona identicamente ao original!")
    else:
        print("✗ ERRO: Sistemas produzem resultados diferentes!")


def exemplo_6_importar_json():
    """
    Exemplo 6: Importar de JSON (mais rico em metadados)
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Importação de JSON")
    print("=" * 70)

    # Cria sistema e importa de JSON
    json_system = fs.MamdaniSystem()

    # Precisa ter variáveis definidas
    json_system.add_input('temperatura', (15, 35))
    json_system.add_input('umidade', (20, 90))
    json_system.add_output('ventilador', (0, 100))

    json_system.add_term('temperatura', 'fria', 'trapezoidal', (15, 15, 18, 22))
    json_system.add_term('temperatura', 'agradavel', 'triangular', (20, 24, 28))
    json_system.add_term('temperatura', 'quente', 'trapezoidal', (26, 30, 35, 35))

    json_system.add_term('umidade', 'seca', 'triangular', (20, 20, 50))
    json_system.add_term('umidade', 'normal', 'triangular', (40, 60, 80))
    json_system.add_term('umidade', 'umida', 'triangular', (70, 90, 90))

    json_system.add_term('ventilador', 'desligado', 'triangular', (0, 0, 20))
    json_system.add_term('ventilador', 'baixo', 'triangular', (10, 30, 50))
    json_system.add_term('ventilador', 'medio', 'triangular', (40, 60, 80))
    json_system.add_term('ventilador', 'alto', 'triangular', (70, 100, 100))

    json_file = os.path.join(os.path.dirname(__file__), 'exported_rules', 'regras_ac.json')

    print(f"\nImportando de JSON: {json_file}")
    json_system.import_rules(json_file)

    print(f"\nSistema importado de JSON:")
    print(f"  Nome: {json_system.name}")
    print(f"  Tipo: {type(json_system).__name__}")
    print(f"  Regras: {len(json_system.rule_base.rules)}")

    print("\nRegras importadas:")
    json_system.print_rules(style='if-then', show_stats=False)


def exemplo_7_uso_pratico():
    """
    Exemplo 7: Caso de uso prático - Compartilhar regras entre exemplos
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 7: Caso de Uso Prático")
    print("=" * 70)

    print("""
CENÁRIO: Você criou um conjunto de regras bem testado para controle
de ar condicionado e quer reutilizar essas regras em outros exemplos.

SOLUÇÃO: Exportar as regras e importá-las quando necessário!
    """)

    # 1. Sistema de referência (já criado anteriormente)
    print("\n1. CRIANDO SISTEMA DE REFERÊNCIA:")
    print("-" * 70)

    reference = fs.MamdaniSystem(name="AC Referência")
    reference.add_input('temp', (15, 35))
    reference.add_output('vent', (0, 100))

    reference.add_term('temp', 'baixa', 'triangular', (15, 15, 22))
    reference.add_term('temp', 'media', 'triangular', (18, 25, 32))
    reference.add_term('temp', 'alta', 'triangular', (28, 35, 35))

    reference.add_term('vent', 'lento', 'triangular', (0, 0, 40))
    reference.add_term('vent', 'medio', 'triangular', (30, 50, 70))
    reference.add_term('vent', 'rapido', 'triangular', (60, 100, 100))

    reference.add_rules([
        (['baixa'], ['lento']),
        (['media'], ['medio']),
        (['alta'], ['rapido']),
    ])

    print("   ✓ Sistema criado com 3 regras")

    # 2. Exportar
    export_file = os.path.join(os.path.dirname(__file__), 'exported_rules', 'regras_padrao.json')
    print(f"\n2. EXPORTANDO REGRAS PADRÃO:")
    print(f"   {export_file}")
    reference.export_rules(export_file)
    print("   ✓ Regras exportadas")

    # 3. Em outro exemplo, importar
    print("\n3. EM OUTRO EXEMPLO, IMPORTAR REGRAS:")
    print("-" * 70)

    novo_exemplo = fs.MamdaniSystem(name="Exemplo Novo")
    novo_exemplo.add_input('temp', (15, 35))
    novo_exemplo.add_output('vent', (0, 100))

    novo_exemplo.add_term('temp', 'baixa', 'triangular', (15, 15, 22))
    novo_exemplo.add_term('temp', 'media', 'triangular', (18, 25, 32))
    novo_exemplo.add_term('temp', 'alta', 'triangular', (28, 35, 35))

    novo_exemplo.add_term('vent', 'lento', 'triangular', (0, 0, 40))
    novo_exemplo.add_term('vent', 'medio', 'triangular', (30, 50, 70))
    novo_exemplo.add_term('vent', 'rapido', 'triangular', (60, 100, 100))

    novo_exemplo.import_rules(export_file)
    print("   ✓ Regras importadas com sucesso!")

    # 4. Usar imediatamente
    print("\n4. USAR SISTEMA IMPORTADO:")
    print("-" * 70)
    result = novo_exemplo.compute({'temp': 30})
    print(f"   Entrada: temp = 30°C")
    print(f"   Saída: vent = {result['vent']:.1f}%")
    print("\n   ✓ Sistema funcionando perfeitamente!")


def main():
    print("\n")
    print("*" * 70)
    print(" " * 15 + "IMPORTAÇÃO E EXPORTAÇÃO DE REGRAS")
    print("*" * 70)

    # Exemplo 1: Visualização
    exemplo_1_visualizar_regras()

    input("\n\nPressione Enter para continuar para o Exemplo 2...")

    # Exemplo 2: DataFrame
    system = exemplo_2_dataframe()

    input("\n\nPressione Enter para continuar para o Exemplo 3...")

    # Exemplo 3: Exportação
    export_dir = exemplo_3_exportar_regras(system)

    input("\n\nPressione Enter para continuar para o Exemplo 4...")

    # Exemplo 4: Importação
    imported_system = exemplo_4_importar_regras(export_dir)

    input("\n\nPressione Enter para continuar para o Exemplo 5...")

    # Exemplo 5: Teste
    exemplo_5_testar_sistema_importado(system, imported_system)

    input("\n\nPressione Enter para continuar para o Exemplo 6...")

    # Exemplo 6: JSON
    exemplo_6_importar_json()

    input("\n\nPressione Enter para continuar para o Exemplo 7...")

    # Exemplo 7: Uso prático
    exemplo_7_uso_pratico()

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DAS FUNCIONALIDADES")
    print("=" * 70)

    print("""
✅ FUNCIONALIDADES IMPLEMENTADAS:

1. VISUALIZAÇÃO DE REGRAS:
   • system.print_rules(style='table')     → Tabela formatada
   • system.print_rules(style='compact')   → Formato compacto
   • system.print_rules(style='detailed')  → Detalhes completos
   • system.print_rules(style='if-then')   → Linguagem natural

2. CONVERSÃO PARA DATAFRAME:
   • system.rules_to_dataframe()                    → Formato compacto
   • system.rules_to_dataframe(expand_variables=True) → Formato expandido

3. ESTATÍSTICAS:
   • system.rules_statistics()  → Total, operadores, médias, etc.

4. EXPORTAÇÃO:
   • system.export_rules('file.csv')   → CSV
   • system.export_rules('file.json')  → JSON (com metadados)
   • system.export_rules('file.txt')   → Texto legível
   • Auto-detecção de formato pela extensão

5. IMPORTAÇÃO:
   • system.import_rules('file.csv')   → Importa de CSV
   • system.import_rules('file.json')  → Importa de JSON
   • Preserva todos os atributos das regras
   • Detecta automaticamente formato compacto/expandido

💡 CASOS DE USO:

• Documentação: Exportar regras em TXT para relatórios
• Backup: Salvar regras em JSON com metadados completos
• Análise: Converter para DataFrame e usar Pandas
• Compartilhamento: Reutilizar regras entre exemplos
• Versionamento: Versionar regras separadamente do código
• Colaboração: Trocar regras com equipe em formatos padronizados

🎯 BOAS PRÁTICAS:

• Use JSON para máxima compatibilidade e metadados
• Use CSV para integração com Excel/planilhas
• Use TXT para documentação legível
• Sempre teste o sistema após importar regras
• Mantenha regras bem testadas em arquivos separados
    """)

    print("\n" + "=" * 70)
    print("FIM DO EXEMPLO")
    print("=" * 70)


if __name__ == "__main__":
    main()
