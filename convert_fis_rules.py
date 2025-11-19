#!/usr/bin/env python3
"""
Script para converter regras do arquivo ASFALTO.fis para JSON
"""
import json

# Mapeamento dos índices para nomes dos termos
UMIDADE_TERMS = ["SECO", "POUCO-UMIDO", "UMIDO", "MUITO-UMIDO", "COMP-MOLHADO"]
TEXTURA_TERMS = ["MUITO-LISA", "LISA", "NORMAL", "RUGOSA", "MUITO-RUGOSA"]
SUJEIRA_TERMS = ["LIMPO", "POUCO-SUJO", "SUJO", "MUITO-SUJO"]
PNEU_TERMS = ["COMP-CARECA", "CARECA", "REGULAR", "BOM", "NOVO"]
CA_TERMS = ["MUITO-BAIXO", "BAIXO", "MEDIO", "ALTO", "MUITO-ALTO"]

# Ler o arquivo .fis
with open('ASFALTO.fis', 'r') as f:
    lines = f.readlines()

# Encontrar a seção de regras
rules = []
in_rules_section = False

for line in lines:
    line = line.strip()

    if line == '[Rules]':
        in_rules_section = True
        continue

    if in_rules_section and line and not line.startswith('['):
        # Parse da regra: "1 1 1 1, 3 (1) : 1"
        # Formato: input1 input2 input3 input4, output (weight) : operator
        parts = line.split(',')
        if len(parts) == 2:
            # Antecedentes
            antecedents_str = parts[0].strip().split()
            umidade_idx = int(antecedents_str[0]) - 1
            textura_idx = int(antecedents_str[1]) - 1
            sujeira_idx = int(antecedents_str[2]) - 1
            pneu_idx = int(antecedents_str[3]) - 1

            # Consequente e peso
            consequent_part = parts[1].strip()
            # Remove o ": 1" do final (operador AND)
            consequent_part = consequent_part.split(':')[0].strip()
            # Parse "3 (1)"
            ca_idx = int(consequent_part.split('(')[0].strip()) - 1
            weight = float(consequent_part.split('(')[1].split(')')[0])

            # Criar regra no formato JSON
            rule = {
                "antecedents": {
                    "UMIDADE": UMIDADE_TERMS[umidade_idx],
                    "TEXTURA": TEXTURA_TERMS[textura_idx],
                    "SUJEIRA": SUJEIRA_TERMS[sujeira_idx],
                    "PNEU": PNEU_TERMS[pneu_idx]
                },
                "consequents": {
                    "CA": CA_TERMS[ca_idx]
                },
                "operator": "AND",
                "weight": weight
            }
            rules.append(rule)

print(f"Total de regras convertidas: {len(rules)}")

# Carregar o JSON atual
with open('ASFALTO.json', 'r') as f:
    data = json.load(f)

# Adicionar as regras
data['rules'] = rules

# Salvar o JSON atualizado
with open('ASFALTO.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Arquivo ASFALTO.json atualizado com 500 regras!")
