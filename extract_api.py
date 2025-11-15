#!/usr/bin/env python3
"""
Script to extract all fuzzy_systems classes and methods used in notebooks.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_imports_from_notebook(notebook_path):
    """Extract fuzzy_systems imports from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    imports = set()
    classes_used = set()
    methods_used = defaultdict(set)

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))

            # Extract imports: from fuzzy_systems.X import Y
            from_imports = re.findall(
                r'from\s+fuzzy_systems(?:\.(\w+))?\s+import\s+([^\n]+)',
                source
            )
            for module, items in from_imports:
                module = module or 'root'
                for item in items.split(','):
                    item = item.strip().split(' as ')[0].strip()
                    if item and not item.startswith('('):
                        imports.add(f"fuzzy_systems.{module}.{item}" if module != 'root' else f"fuzzy_systems.{item}")
                        classes_used.add(item)

            # Extract direct imports: import fuzzy_systems as fs
            direct_imports = re.findall(r'import\s+fuzzy_systems(?:\s+as\s+(\w+))?', source)
            if direct_imports:
                imports.add('fuzzy_systems')

            # Extract class instantiations and method calls
            # Pattern: ClassName(...) or obj.method_name(...)
            class_instantiations = re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\s*\(', source)
            for cls in class_instantiations:
                classes_used.add(cls)

            # Extract method calls: obj.method(...)
            method_calls = re.findall(r'\.([a-z_][a-z0-9_]*)\s*\(', source)
            for method in method_calls:
                if not method.startswith('_'):  # Skip private methods
                    methods_used['general'].add(method)

    return imports, classes_used, dict(methods_used)


def analyze_all_notebooks(notebooks_dir):
    """Analyze all notebooks in the directory."""
    notebooks_dir = Path(notebooks_dir)

    all_imports = set()
    all_classes = set()
    all_methods = defaultdict(set)
    notebook_map = {}

    for notebook_path in notebooks_dir.rglob('*.ipynb'):
        if '.ipynb_checkpoints' in str(notebook_path):
            continue

        imports, classes, methods = extract_imports_from_notebook(notebook_path)

        all_imports.update(imports)
        all_classes.update(classes)
        for key, method_set in methods.items():
            all_methods[key].update(method_set)

        notebook_map[notebook_path.relative_to(notebooks_dir)] = {
            'imports': list(imports),
            'classes': list(classes)
        }

    return all_imports, all_classes, dict(all_methods), notebook_map


def organize_by_module(all_classes):
    """Organize classes by module based on common patterns."""
    modules = {
        'core': ['FuzzySet', 'LinguisticVariable', 'triangular', 'trapezoidal',
                 'gaussian', 'sigmoid', 'generalized_bell', 'fuzzy_and_min',
                 'fuzzy_or_max', 'fuzzy_not'],
        'inference': ['MamdaniSystem', 'SugenoSystem'],
        'learning': ['WangMendelLearning', 'ANFIS', 'MamdaniLearning', 'WangMendel'],
        'dynamics': ['PFuzzyDiscrete', 'PFuzzyContinuous', 'FuzzyODE']
    }

    organized = defaultdict(list)
    for cls in all_classes:
        for module, module_classes in modules.items():
            if cls in module_classes:
                organized[module].append(cls)
                break
        else:
            organized['other'].append(cls)

    return dict(organized)


def main():
    notebooks_dir = Path(__file__).parent.parent / 'notebooks_colab'

    print("🔍 Analyzing notebooks...")
    all_imports, all_classes, all_methods, notebook_map = analyze_all_notebooks(notebooks_dir)

    print("\n📦 Imports found:")
    for imp in sorted(all_imports):
        print(f"  - {imp}")

    print(f"\n🎯 Classes found ({len(all_classes)}):")
    organized = organize_by_module(all_classes)
    for module, classes in sorted(organized.items()):
        if classes:
            print(f"\n  {module}:")
            for cls in sorted(classes):
                print(f"    - {cls}")

    print(f"\n🔧 Methods found ({len(all_methods.get('general', []))}):")
    for method in sorted(all_methods.get('general', []))[:20]:  # Show first 20
        print(f"  - {method}")

    # Save results
    output = {
        'imports': sorted(all_imports),
        'classes_by_module': {k: sorted(v) for k, v in organized.items()},
        'common_methods': sorted(all_methods.get('general', []))[:50],
        'notebook_analysis': {str(k): v for k, v in notebook_map.items()}
    }

    output_path = Path(__file__).parent / 'api_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Analysis saved to: {output_path}")
    return output


if __name__ == '__main__':
    main()
