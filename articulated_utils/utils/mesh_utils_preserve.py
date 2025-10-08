"""
完全保持原始位置的mesh处理函数
"""
import numpy as np
import shutil
import os
from .mesh_utils import load_obj, save_obj, get_aabb


def add_mesh_preserve_original_position(box_spec, mesh_dir):
    """
    完全保持原始mesh的位置和尺寸，只做最基本的文件复制
    """
    try:
        # 读取原始文件信息用于更新box_spec
        vertices, faces = load_obj(mesh_dir)
        min_coords, max_coords = get_aabb(vertices)
        mesh_extent = max_coords - min_coords
        
        # 更新box_spec但不修改顶点
        box_spec['x'] = float(mesh_extent[0])
        box_spec['y'] = float(mesh_extent[1])
        box_spec['z'] = float(mesh_extent[2])
        
        # 返回原始的未修改的顶点和面
        return vertices, faces
        
    except Exception as e:
        print(f"Error in add_mesh_preserve_original_position: {e}")
        # 如果出错，回退到读取原始文件
        vertices, faces = load_obj(mesh_dir)
        return vertices, faces


def copy_mesh_preserve_position(source_path, target_path):
    """
    直接复制mesh文件，不做任何修改
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 直接复制文件
        shutil.copy2(source_path, target_path)
        
        print(f"Successfully copied {source_path} to {target_path}")
        return True
        
    except Exception as e:
        print(f"Error copying mesh: {e}")
        return False