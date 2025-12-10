#!/usr/bin/env python3
"""Patch image_mesh.py to make utils3d.numpy optional."""
import sys
import os

file_path = sys.argv[1] if len(sys.argv) > 1 else 'sam3d_objects/utils/visualization/image_mesh.py'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(0)

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    if 'from utils3d.numpy import' in line:
        new_lines.append('try:\n')
        new_lines.append('    from utils3d.numpy import (\n')
        i += 1
        # Copy import lines
        while i < len(lines) and ')' not in lines[i]:
            new_lines.append('    ' + lines[i])
            i += 1
        new_lines.append('    )\n')
        new_lines.append('    Utils3dNumpyAvailable = True\n')
        new_lines.append('except ImportError:\n')
        new_lines.append('    Utils3dNumpyAvailable = False\n')
        new_lines.append('    def depth_edge(*args, **kwargs):\n')
        new_lines.append('        raise NotImplementedError("utils3d.numpy.depth_edge not available")\n')
        new_lines.append('    def normals_edge(*args, **kwargs):\n')
        new_lines.append('        raise NotImplementedError("utils3d.numpy.normals_edge not available")\n')
        new_lines.append('    def points_to_normals(*args, **kwargs):\n')
        new_lines.append('        raise NotImplementedError("utils3d.numpy.points_to_normals not available")\n')
        new_lines.append('    def image_uv(*args, **kwargs):\n')
        new_lines.append('        raise NotImplementedError("utils3d.numpy.image_uv not available")\n')
        new_lines.append('    def image_mesh(*args, **kwargs):\n')
        new_lines.append('        raise NotImplementedError("utils3d.numpy.image_mesh not available")\n')
        i += 1
        continue
    else:
        new_lines.append(line)
        i += 1

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("Patched image_mesh.py to make utils3d.numpy optional")

