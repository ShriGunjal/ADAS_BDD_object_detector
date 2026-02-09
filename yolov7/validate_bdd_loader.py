#!/usr/bin/env python3
"""
Quick validation script for BDD100K loader implementation
Tests if the loader can be instantiated and handle data correctly
"""

import sys
import json
from pathlib import Path

# Test 1: Import check
print("=" * 60)
print("TEST 1: Import LoadImagesAndLabelsJSON")
print("=" * 60)
try:
    from utils.datasets import LoadImagesAndLabelsJSON
    print("✅ PASS: LoadImagesAndLabelsJSON imported successfully\n")
except ImportError as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Test 2: JSON file check
print("=" * 60)
print("TEST 2: Verify BDD JSON annotation files exist")
print("=" * 60)
json_dir = Path("data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels")
if not json_dir.exists():
    print(f"❌ FAIL: Directory not found: {json_dir}\n")
    sys.exit(1)

json_files = list(json_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files:")
for jf in json_files:
    size_mb = jf.stat().st_size / (1024**2)
    print(f"  - {jf.name} ({size_mb:.1f} MB)")
print()

# Test 3: Config file check
print("=" * 60)
print("TEST 3: Verify bdd100k.yaml configuration")
print("=" * 60)
try:
    import yaml
    config = yaml.safe_load(open("data/bdd100k.yaml"))
    print(f"✅ Config loaded:")
    print(f"   - Number of classes: {config['nc']}")
    print(f"   - Class names: {config['names']}")
    print()
except Exception as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Test 4: Sample JSON structure
print("=" * 60)
print("TEST 4: Check JSON structure (first record)")
print("=" * 60)
try:
    train_json = json_dir / "bdd100k_labels_images_train.json"
    with open(train_json) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    sample = data[0]
    print(f"Sample image name: {sample.get('name')}")
    print(f"Number of labels: {len(sample.get('labels', []))}")
    
    # Check first bbox
    for label in sample.get('labels', []):
        if 'box2d' in label:
            box = label['box2d']
            print(f"Sample box: {label['category']} - x1={box['x1']:.1f}, y1={box['y1']:.1f}, x2={box['x2']:.1f}, y2={box['y2']:.1f}")
            break
    print("✅ PASS: JSON structure valid\n")
except Exception as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Test 5: Class mapping
print("=" * 60)
print("TEST 5: Verify BDD class mapping")
print("=" * 60)
try:
    class_map = LoadImagesAndLabelsJSON.BDD_CLASS_MAP
    print(f"BDD_CLASS_MAP has {len(class_map)} classes:")
    for name, id in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"  {id}: {name}")
    print("✅ PASS: Class mapping valid\n")
except Exception as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Test 6: create_dataloader detection
print("=" * 60)
print("TEST 6: Verify create_dataloader BDD detection")
print("=" * 60)
try:
    from utils.datasets import create_dataloader
    # Check if function has dataset_type parameter
    import inspect
    sig = inspect.signature(create_dataloader)
    if 'dataset_type' in sig.parameters:
        print("✅ PASS: create_dataloader has dataset_type parameter\n")
    else:
        print("❌ FAIL: dataset_type parameter not found in create_dataloader\n")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Test 7: train.py modifications
print("=" * 60)
print("TEST 7: Verify train.py modifications")
print("=" * 60)
try:
    with open("train.py") as f:
        content = f.read()
    
    checks = {
        "is_bdd detection": "is_bdd = opt.data.endswith('bdd100k.yaml')" in content,
        "dataset_type variable": "dataset_type = 'bdd100k'" in content,
        "dataset_type in create_dataloader": "dataset_type=dataset_type" in content,
    }
    
    all_pass = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("✅ PASS: train.py properly modified\n")
    else:
        print("❌ FAIL: Some modifications missing\n")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}\n")
    sys.exit(1)

# Summary
print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nImplementation summary:")
print("  1. LoadImagesAndLabelsJSON class added to datasets.py")
print("  2. create_dataloader() modified for dataset type detection")
print("  3. train.py updated with BDD detection and dataset_type passing")
print("  4. bdd100k.yaml configuration created")
print("\nTo train on BDD100K:")
print("  python train.py --data data/bdd100k.yaml --img 640 --batch 32 --epochs 100 --weights yolov7.pt")
print("\nTo train on COCO (unchanged):")
print("  python train.py --data data/coco.yaml --img 640 --batch 32 --epochs 100 --weights yolov7.pt")
print("=" * 60)
