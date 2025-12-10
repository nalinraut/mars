#!/usr/bin/env python3
"""
Patch pointmap.py to gracefully handle moge imports.

MoGe is now properly installed, but this patch ensures graceful
degradation if there are any import issues with specific submodules.
"""
import sys
import os

file_path = sys.argv[1] if len(sys.argv) > 1 else 'sam3d_objects/pipeline/utils/pointmap.py'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(0)

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if 'MOGE_AVAILABLE' in content:
    print(f"Already patched: {file_path}")
    sys.exit(0)

# Find and replace the moge import section
old_import = '''# Import directly from moge
from moge.utils.geometry_torch import (
    normalized_view_plane_uv,
    recover_focal_shift,
)
from moge.utils.geometry_numpy import (
    solve_optimal_focal_shift,
    solve_optimal_shift,
)'''

new_import = '''# Import MoGe utilities with graceful fallback
MOGE_AVAILABLE = False
try:
    from moge.utils.geometry_torch import (
        normalized_view_plane_uv,
        recover_focal_shift,
    )
    from moge.utils.geometry_numpy import (
        solve_optimal_focal_shift,
        solve_optimal_shift,
    )
    MOGE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"MoGe geometry utilities not available: {e}")
    # Stub functions for graceful degradation
    def normalized_view_plane_uv(*args, **kwargs):
        raise NotImplementedError("MoGe geometry_torch not available")
    def recover_focal_shift(*args, **kwargs):
        raise NotImplementedError("MoGe geometry_torch not available")
    def solve_optimal_focal_shift(*args, **kwargs):
        raise NotImplementedError("MoGe geometry_numpy not available")
    def solve_optimal_shift(*args, **kwargs):
        raise NotImplementedError("MoGe geometry_numpy not available")'''

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Patched moge imports in: {file_path}")
else:
    # Try alternate patterns (broken imports from previous patches)
    patterns_to_find = [
        '# Import directly from moge',
        'from moge.utils.geometry_torch import',
        'from moge.utils.geometry_numpy import',
    ]
    
    found = any(p in content for p in patterns_to_find)
    if found and 'MOGE_AVAILABLE' not in content:
        # More aggressive replacement - find the import section
        lines = content.split('\n')
        new_lines = []
        skip_until_def = False
        inserted = False
        
        for i, line in enumerate(lines):
            if '# Import directly from moge' in line and not inserted:
                # Insert our new import block
                new_lines.append(new_import)
                skip_until_def = True
                inserted = True
                continue
            
            if skip_until_def:
                # Skip old moge import lines
                if line.strip().startswith('from moge') or \
                   line.strip().startswith(')') or \
                   line.strip() == '' or \
                   'solve_optimal' in line or \
                   'normalized_view' in line or \
                   'recover_focal' in line:
                    continue
                else:
                    skip_until_def = False
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Patched moge imports (alternate pattern) in: {file_path}")
    else:
        print(f"No moge import pattern found or already patched in: {file_path}")
