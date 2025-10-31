# Obsolete Files

This folder contains files that are not currently used in the public API but are kept for reference and potential future use.

## Learning Module

### Files Moved (2025-10-31)

| File | Reason | Status |
|------|--------|--------|
| `anfis_bk.py` | Backup of ANFIS implementation | Replaced by `anfis.py` |
| `anfis_original_backup.py` | Original ANFIS backup | Replaced by `anfis.py` |
| `adaptive_wang_mendel.py` | Adaptive Wang-Mendel variant | Not used in public API |
| `mamdani.py` | Alternative Mamdani learning implementation | Replaced by `mandani_learning.py` |

### Why These Files Were Moved

1. **anfis_bk.py** and **anfis_original_backup.py**:
   - These are backup versions created during ANFIS development
   - The current `anfis.py` (104KB) is the final, optimized version
   - Kept for historical reference only

2. **adaptive_wang_mendel.py**:
   - Implementation of adaptive Wang-Mendel with iterative optimization
   - Not imported in `learning/__init__.py`
   - Not used in any notebooks
   - Can be resurrected if needed in future versions

3. **mamdani.py**:
   - Alternative Mamdani learning implementation
   - Functionality merged into `mandani_learning.py`
   - Not imported in `learning/__init__.py`
   - Kept for reference

## Restoration

If any of these files are needed:
1. Copy the file back to its original location in `fuzzy_systems/learning/`
2. Add the import to `fuzzy_systems/learning/__init__.py`
3. Update the `__all__` list

## Deletion Policy

These files may be permanently deleted in a future major version (2.0.0+) if they remain unused.

---

**Note:** All files in this folder were moved after analysis showing:
- They are not imported in any `__init__.py`
- They are not used in any notebooks
- They are not referenced in the documentation
