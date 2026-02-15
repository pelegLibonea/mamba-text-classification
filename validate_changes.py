"""
Simple validation script to check the code changes are syntactically correct.
This doesn't run the model but validates the changes.
"""

import ast
import sys

def validate_python_file(filepath):
    """Check if a Python file is syntactically correct."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_function_exists(filepath, function_name):
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return True
        return False
    except:
        return False

def check_class_method_exists(filepath, class_name, method_name):
    """Check if a class method exists."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return True
        return False
    except:
        return False

def check_parameter_exists(filepath, class_name, method_name, param_name):
    """Check if a parameter exists in a method."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        # Check args
                        for arg in item.args.args:
                            if arg.arg == param_name:
                                return True
                        # Check kwonlyargs
                        for arg in item.args.kwonlyargs:
                            if arg.arg == param_name:
                                return True
        return False
    except:
        return False

print("=" * 60)
print("VALIDATION CHECKS FOR MULTI-CLASS CLASSIFICATION CHANGES")
print("=" * 60)

all_passed = True

# Check 1: Syntax validation
print("\n1. Checking syntax of modified files...")
files_to_check = [
    "mamba/model.py",
    "dataset.py",
    "utils.py",
    "trainer.py",
]

for filepath in files_to_check:
    valid, msg = validate_python_file(filepath)
    status = "✓" if valid else "✗"
    print(f"   {status} {filepath}: {msg}")
    if not valid:
        all_passed = False

# Check 2: Model changes
print("\n2. Checking MambaTextClassification changes...")

# Check num_classes parameter in __init__
has_param = check_parameter_exists("mamba/model.py", "MambaTextClassification", "__init__", "num_classes")
status = "✓" if has_param else "✗"
print(f"   {status} __init__ has num_classes parameter: {has_param}")
if not has_param:
    all_passed = False

# Check save_pretrained method
has_save = check_class_method_exists("mamba/model.py", "MambaTextClassification", "save_pretrained")
status = "✓" if has_save else "✗"
print(f"   {status} save_pretrained method exists: {has_save}")
if not has_save:
    all_passed = False

# Check load_pretrained_local method
has_load = check_class_method_exists("mamba/model.py", "MambaTextClassification", "load_pretrained_local")
status = "✓" if has_load else "✗"
print(f"   {status} load_pretrained_local method exists: {has_load}")
if not has_load:
    all_passed = False

# Check from_pretrained has num_classes
has_param = check_parameter_exists("mamba/model.py", "MambaTextClassification", "from_pretrained", "num_classes")
status = "✓" if has_param else "✗"
print(f"   {status} from_pretrained has num_classes parameter: {has_param}")
if not has_param:
    all_passed = False

# Check 3: Dataset changes
print("\n3. Checking dataset changes...")
has_multiclass = check_class_method_exists("dataset.py", "MultiClassDataset", "__init__")
status = "✓" if has_multiclass else "✗"
print(f"   {status} MultiClassDataset class exists: {has_multiclass}")
if not has_multiclass:
    all_passed = False

# Check 4: Utils changes
print("\n4. Checking utils.py changes...")
has_compute_metrics = check_function_exists("utils.py", "compute_metrics")
status = "✓" if has_compute_metrics else "✗"
print(f"   {status} compute_metrics function exists: {has_compute_metrics}")
if not has_compute_metrics:
    all_passed = False

# Check 5: Requirements
print("\n5. Checking requirements.txt...")
with open("requirements.txt", "r") as f:
    requirements = f.read()
has_sklearn = "scikit-learn" in requirements
status = "✓" if has_sklearn else "✗"
print(f"   {status} scikit-learn added to requirements: {has_sklearn}")
if not has_sklearn:
    all_passed = False

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("=" * 60)
    sys.exit(0)
else:
    print("❌ SOME VALIDATION CHECKS FAILED!")
    print("=" * 60)
    sys.exit(1)
