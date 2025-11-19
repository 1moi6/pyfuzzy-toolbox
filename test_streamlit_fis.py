#!/usr/bin/env python3
"""
Test script to verify .fis file can be loaded and converted to internal format
"""
import tempfile
import os
from fuzzy_systems.inference import MamdaniSystem

print("=" * 60)
print("Testing FIS file loading for Streamlit")
print("=" * 60)

# Simulate uploaded file by reading ASFALTO.fis
with open('ASFALTO.fis', 'r') as f:
    file_content = f.read()

print("\n📥 Simulating file upload...")

# Save to temporary file (simulating Streamlit upload)
with tempfile.NamedTemporaryFile(mode='w', suffix='.fis', delete=False) as tmp_file:
    tmp_file.write(file_content)
    tmp_path = tmp_file.name

try:
    print(f"   Temp file: {tmp_path}")

    # Load using from_fis
    print("\n🔄 Loading FIS file...")
    fis_system = MamdaniSystem.from_fis(tmp_path)

    # Convert to JSON format (as done in Streamlit)
    print("\n🔄 Converting to internal JSON format...")
    json_str = fis_system.to_json()
    import json
    json_data = json.loads(json_str)

    print("\n✅ Conversion successful!")
    print(f"   Name: {json_data.get('name', 'Unnamed')}")
    print(f"   Type: {json_data.get('system_type', 'Unknown')}")
    print(f"   Inputs: {len(json_data.get('input_variables', {}))}")
    print(f"   Outputs: {len(json_data.get('output_variables', {}))}")
    print(f"   Rules: {len(json_data.get('rules', []))}")

    # Test that we can access input variables
    print("\n📊 Input variables:")
    for var_name, var_data in json_data.get('input_variables', {}).items():
        universe = var_data.get('universe', [0, 100])
        terms = var_data.get('terms', {})
        print(f"   • {var_name}: [{universe[0]}, {universe[-1]}] - {len(terms)} terms")

    print("\n" + "=" * 60)
    print("✅ Streamlit FIS upload functionality will work!")
    print("=" * 60)

finally:
    # Clean up temp file
    os.unlink(tmp_path)
    print(f"\n🗑️  Cleaned up temp file")
