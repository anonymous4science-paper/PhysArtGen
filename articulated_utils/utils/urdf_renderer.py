"""
URDF渲染工具 - 支持多种渲染后端
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class BaseRenderer:
    """渲染器基类"""
    
    def __init__(self, resolution: List[int] = [512, 512]):
        self.resolution = resolution
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def render_urdf(self, urdf_path: str, output_path: str, 
                   camera_pos: List[float], camera_target: List[float]) -> bool:
        """渲染URDF文件"""
        raise NotImplementedError


class SapienRenderer(BaseRenderer):
    """SAPIEN渲染器实现"""
    
    def __init__(self, resolution: List[int] = [512, 512]):
        super().__init__(resolution)
        self.engine = None
        self.scene = None
        
    def _init_engine(self):
        """初始化SAPIEN引擎"""
        try:
            import sapien.core as sapien
            
            self.engine = sapien.Engine()
            renderer = sapien.SapienRenderer()
            self.engine.set_renderer(renderer)
            
            # 创建场景
            scene_config = sapien.SceneConfig()
            self.scene = self.engine.create_scene(scene_config)
            
            # 设置环境光照
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)
            
            return True
            
        except ImportError:
            self.logger.error("SAPIEN库未安装")
            return False
        except Exception as e:
            self.logger.error(f"初始化SAPIEN失败: {e}")
            return False
    
    def render_urdf(self, urdf_path: str, output_path: str,
                   camera_pos: List[float], camera_target: List[float]) -> bool:
        """使用SAPIEN渲染URDF"""
        if not self.engine and not self._init_engine():
            return False
        
        try:
            # 加载URDF
            loader = self.scene.create_urdf_loader()
            robot = loader.load(urdf_path)
            
            if robot is None:
                self.logger.error(f"无法加载URDF: {urdf_path}")
                return False
            
            # 设置相机
            camera = self.scene.add_camera(
                name="render_camera",
                width=self.resolution[0],
                height=self.resolution[1],
                fovy=np.deg2rad(35),
                near=0.1,
                far=100
            )
            
            # 设置相机位置和朝向
            camera_pose = self._look_at(camera_pos, camera_target, [0, 0, 1])
            camera.set_pose(camera_pose)
            
            # 渲染
            self.scene.step()  # 物理步进
            self.scene.update_render()
            camera.take_picture()
            
            # 获取图像数据
            rgba = camera.get_color_rgba()  # [H, W, 4]
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            
            # 保存图像
            import cv2
            cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # 清理
            self.scene.remove_actor(robot)
            self.scene.remove_camera(camera)
            
            return True
            
        except Exception as e:
            self.logger.error(f"SAPIEN渲染失败: {e}")
            return False
    
    def _look_at(self, eye: List[float], target: List[float], up: List[float]):
        """计算相机姿态"""
        import sapien.core as sapien
        
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)
        
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        rotation_matrix = np.column_stack([right, up, -forward])
        pose = sapien.Pose(eye, rotation_matrix)
        
        return pose


class PybulletRenderer(BaseRenderer):
    """PyBullet渲染器实现"""
    
    def __init__(self, resolution: List[int] = [512, 512]):
        super().__init__(resolution)
        self.physics_client = None
    
    def _init_pybullet(self):
        """初始化PyBullet"""
        try:
            import pybullet as p
            
            # 连接到PyBullet
            self.physics_client = p.connect(p.DIRECT)  # 无GUI模式
            
            # 设置重力
            p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
            
            return True
            
        except ImportError:
            self.logger.error("PyBullet库未安装")
            return False
        except Exception as e:
            self.logger.error(f"初始化PyBullet失败: {e}")
            return False
    
    def render_urdf(self, urdf_path: str, output_path: str,
                   camera_pos: List[float], camera_target: List[float]) -> bool:
        """使用PyBullet渲染URDF"""
        if not self.physics_client and not self._init_pybullet():
            return False
        
        try:
            import pybullet as p
            
            # 加载URDF
            robot_id = p.loadURDF(urdf_path, physicsClientId=self.physics_client)
            
            if robot_id < 0:
                self.logger.error(f"无法加载URDF: {urdf_path}")
                return False
            
            # 设置相机参数
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=camera_target,
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics_client
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=35,  # 视场角
                aspect=self.resolution[0] / self.resolution[1],
                nearVal=0.1,
                farVal=100,
                physicsClientId=self.physics_client
            )
            
            # 渲染
            width, height, rgb_array, depth_array, seg_array = p.getCameraImage(
                width=self.resolution[0],
                height=self.resolution[1],
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                physicsClientId=self.physics_client
            )
            
            # 转换图像格式
            rgb_array = rgb_array[:, :, :3]  # 移除alpha通道
            
            # 保存图像
            import cv2
            cv2.imwrite(output_path, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            
            # 清理
            p.removeBody(robot_id, physicsClientId=self.physics_client)
            
            return True
            
        except Exception as e:
            self.logger.error(f"PyBullet渲染失败: {e}")
            return False
    
    def __del__(self):
        """清理资源"""
        if self.physics_client is not None:
            try:
                import pybullet as p
                p.disconnect(physicsClientId=self.physics_client)
            except:
                pass