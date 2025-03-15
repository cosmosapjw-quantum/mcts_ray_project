# fix_format.py
"""
Small patch to fix the format string error in train_optimized.py
"""

import re

def fix_format_string_error():
    """Fix the format string error in train_optimized.py"""
    with open('train_optimized.py', 'r') as file:
        content = file.read()
    
    # Replace the problematic format string
    fixed_content = content.replace(
        "print(f\"  Loss: {loss:.4f if loss else 'N/A'}\")",
        "print(f\"  Loss: {loss:.4f if loss is not None else 'N/A'}\")"
    )
    
    with open('train_optimized.py', 'w') as file:
        file.write(fixed_content)
    
    print("Fixed format string error in train_optimized.py")

if __name__ == "__main__":
    fix_format_string_error()