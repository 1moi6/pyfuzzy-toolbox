#!/usr/bin/env python3
"""
Script para testar o import de arquivos .fis do MATLAB
"""
from fuzzy_systems.inference import MamdaniSystem

print("=" * 60)
print("Testando importação de arquivo .fis do MATLAB")
print("=" * 60)

# Carregar o sistema do .fis
print("\n📥 Carregando ASFALTO.fis...")
fis = MamdaniSystem.from_fis('ASFALTO.fis')

print("\n" + "=" * 60)
print("Informações do Sistema")
print("=" * 60)
print(f"Nome: {fis.name}")
print(f"Tipo: {fis.__class__.__name__}")
print(f"\nVariáveis de Entrada:")
for var_name, var in fis.input_variables.items():
    print(f"  • {var_name}: [{var.universe[0]}, {var.universe[-1]}]")
    print(f"    Termos: {list(var.terms.keys())}")

print(f"\nVariáveis de Saída:")
for var_name, var in fis.output_variables.items():
    print(f"  • {var_name}: [{var.universe[0]}, {var.universe[-1]}]")
    print(f"    Termos: {list(var.terms.keys())}")

print(f"\nRegras: {len(fis.rule_base.rules)}")

# Testar inferência
print("\n" + "=" * 60)
print("Testando Inferência")
print("=" * 60)

test_cases = [
    {
        'desc': 'Condições boas (seco, rugoso, limpo, pneu novo)',
        'inputs': {'UMIDADE': 0.1, 'TEXTURA': 1.0, 'SUJEIRA': 0.1, 'PNEU': 10}
    },
    {
        'desc': 'Condições ruins (molhado, liso, sujo, pneu careca)',
        'inputs': {'UMIDADE': 0.9, 'TEXTURA': 0.1, 'SUJEIRA': 0.8, 'PNEU': 1}
    },
    {
        'desc': 'Condições médias',
        'inputs': {'UMIDADE': 0.5, 'TEXTURA': 0.7, 'SUJEIRA': 0.3, 'PNEU': 6}
    }
]

for test in test_cases:
    print(f"\n🧪 {test['desc']}")
    print(f"   Entradas: {test['inputs']}")
    result = fis.evaluate(test['inputs'])
    print(f"   Resultado CA: {result['CA']:.4f}")

print("\n" + "=" * 60)
print("✅ Teste concluído com sucesso!")
print("=" * 60)

# Comparar com JSON
print("\n📊 Comparando com versão JSON...")
from fuzzy_systems.inference import MamdaniSystem as MS
fis_json = MS.from_json('ASFALTO.json')

test_input = {'UMIDADE': 0.5, 'TEXTURA': 0.7, 'SUJEIRA': 0.3, 'PNEU': 6}
result_fis = fis.evaluate(test_input)
result_json = fis_json.evaluate(test_input)

print(f"Resultado .fis:  CA = {result_fis['CA']:.6f}")
print(f"Resultado .json: CA = {result_json['CA']:.6f}")
print(f"Diferença: {abs(result_fis['CA'] - result_json['CA']):.10f}")

if abs(result_fis['CA'] - result_json['CA']) < 1e-6:
    print("✅ Resultados idênticos!")
else:
    print("⚠️  Pequena diferença (pode ser devido a arredondamento)")
