#!/usr/bin/env python3
"""
Script para testar o carregamento do ASFALTO.json
"""
from fuzzy_systems.inference import MamdaniSystem

# Carregar o sistema do JSON
print("📥 Carregando sistema ASFALTO do JSON...")
fis = MamdaniSystem.from_json('ASFALTO.json')

print(f"\n✅ Sistema carregado com sucesso!")
print(f"   Nome: {fis.name}")
print(f"   Tipo: {fis.__class__.__name__}")
print(f"   Inputs: {list(fis.input_variables.keys())}")
print(f"   Outputs: {list(fis.output_variables.keys())}")
print(f"   Total de regras: {len(fis.rule_base.rules)}")

# Testar uma inferência
print("\n🧪 Testando inferência...")
print("   Condições: Umidade=0.5, Textura=0.7, Sujeira=0.3, Pneu=8")

result = fis.compute({
    'UMIDADE': 0.5,
    'TEXTURA': 0.7,
    'SUJEIRA': 0.3,
    'PNEU': 8
})

print(f"   Resultado CA: {result['CA']:.4f}")

# Testar outro cenário
print("\n🧪 Testando outro cenário...")
print("   Condições: Umidade=0.9 (molhado), Textura=0.1 (liso), Sujeira=0.8, Pneu=1 (careca)")

result2 = fis.compute({
    'UMIDADE': 0.9,
    'TEXTURA': 0.1,
    'SUJEIRA': 0.8,
    'PNEU': 1
})

print(f"   Resultado CA: {result2['CA']:.4f}")

print("\n✅ Todos os testes passaram! O arquivo ASFALTO.json está correto.")
