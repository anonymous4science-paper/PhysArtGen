"""
修改版的mesh_utils，保持OBJ文件的相对位置关系
"""
import numpy as np
import shutil
import os
from .mesh_utils import load_obj, get_aabb, transform_vertices, save_obj


def add_mesh_preserve_position(box_spec, mesh_dir):
    """
    处理mesh但保持相对位置关系，不进行中心化
    """
    # partnet mobility artifact. Rotate the mesh upright
    up_axis_transform = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    vertices, faces = load_obj(mesh_dir)
    
    # **关键修改：不进行中心化，保持原始位置**
    # 注释掉中心化代码
    # min_coords, max_coords = get_aabb(vertices)
    # center = (min_coords + max_coords) / 2
    # vertices -= center
    
    # Apply up-axis transform (保持坐标系变换，但不移动位置)
    vertices = transform_vertices(vertices, up_axis_transform)
    
    # 保持原始尺寸，不进行缩放
    min_coords, max_coords = get_aabb(vertices)
    mesh_extent = max_coords - min_coords
    
    # 使用原始的边界框信息
    box_spec['x'] = float(mesh_extent[0])
    box_spec['y'] = float(mesh_extent[1]) 
    box_spec['z'] = float(mesh_extent[2])
    
    return vertices, faces


def process_mesh_preserve_position(box_spec, mesh_dir):
    """
    简化版本，直接复制mesh文件而不进行任何变换
    这样可以完全保持原始的相对位置关系
    """
    try:
        # 直接复制原始OBJ文件，不做任何修改
        target_path = mesh_dir.replace('.obj', '_processed.obj')
        shutil.copy2(mesh_dir, target_path)
        
        # 读取原始mesh信息用于box_spec
        vertices, faces = load_obj(mesh_dir)
        min_coords, max_coords = get_aabb(vertices)
        mesh_extent = max_coords - min_coords
        
        # 更新box_spec
        box_spec['x'] = float(mesh_extent[0])
        box_spec['y'] = float(mesh_extent[1])
        box_spec['z'] = float(mesh_extent[2])
        
        return vertices, faces
        
    except Exception as e:
        print(f"Error processing mesh {mesh_dir}: {e}")
        # 如果出错，使用最小变换版本
        return add_mesh_preserve_position(box_spec, mesh_dir)