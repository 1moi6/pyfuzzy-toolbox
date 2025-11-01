"""
Script to simplify Jupyter notebooks by reducing markdown verbosity.

Keeps:
- Main title (first H1)
- Section headers (H2/H3)
- Code cells with inline comments
- Brief descriptions (< 200 chars)

Removes:
- Long explanatory markdown cells
- Redundant text
"""

import json
import glob
import re
from pathlib import Path


def is_header(cell_source):
    """Check if cell is a header (# or ## or ###)."""
    text = ''.join(cell_source).strip()
    return text.startswith('#') and not text.startswith('####')


def is_brief(cell_source):
    """Check if cell is brief (< 100 chars)."""
    text = ''.join(cell_source).strip()
    return len(text) < 100


def is_exercise_or_important(cell_source):
    """Check if cell contains exercise or important marker."""
    text = ''.join(cell_source).strip()
    return '🎯' in text or 'Exercise' in text or 'Exercício' in text


def simplify_notebook(notebook_path):
    """Simplify a single notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    original_cells = len(nb['cells'])
    md_count_before = len([c for c in nb['cells'] if c['cell_type'] == 'markdown'])

    new_cells = []
    seen_title = False

    for cell in nb['cells']:
        # Always keep code cells
        if cell['cell_type'] == 'code':
            new_cells.append(cell)
            continue

        # For markdown cells
        if cell['cell_type'] == 'markdown':
            # Keep first title
            if not seen_title and is_header(cell['source']):
                new_cells.append(cell)
                seen_title = True
                continue

            # Keep section headers (only H2)
            if is_header(cell['source']):
                text = ''.join(cell['source']).strip()
                # Only keep ## headers (not ###)
                if text.startswith('##') and not text.startswith('###'):
                    new_cells.append(cell)
                    continue

            # Keep exercises
            if is_exercise_or_important(cell['source']):
                new_cells.append(cell)
                continue

            # Keep very brief descriptions
            if is_brief(cell['source']):
                new_cells.append(cell)
                continue

    nb['cells'] = new_cells

    md_count_after = len([c for c in nb['cells'] if c['cell_type'] == 'markdown'])

    # Only save if we actually removed cells
    if len(new_cells) < original_cells:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        removed = original_cells - len(new_cells)
        md_removed = md_count_before - md_count_after
        return True, removed, md_removed

    return False, 0, 0


def main():
    """Simplify all notebooks."""
    notebooks = sorted(glob.glob('notebooks_colab/**/*.ipynb', recursive=True))

    print(f"Found {len(notebooks)} notebooks\n")
    print("Simplifying notebooks...\n")

    total_removed = 0
    total_md_removed = 0
    modified_count = 0

    for nb_path in notebooks:
        modified, removed, md_removed = simplify_notebook(nb_path)

        if modified:
            modified_count += 1
            total_removed += removed
            total_md_removed += md_removed
            name = Path(nb_path).name
            print(f"✓ {name:50} removed {removed:2} cells ({md_removed} markdown)")

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Modified notebooks: {modified_count}/{len(notebooks)}")
    print(f"  Total cells removed: {total_removed}")
    print(f"  Markdown cells removed: {total_md_removed}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
