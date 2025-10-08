#!/usr/bin/env python3


import argparse
import os
import sys
from typing import Dict, Optional
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from articulated_utils.physics.sapien_simulate import (
        setup_sapien, get_manipulatable_joints, capture_video, capture_photo
    )
    from articulated_utils.physics.sapien_render import setup_cameras
    from articulated_utils.utils.utils import create_dir
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure SAPIEN and project dependencies are installed")
    sys.exit(1)


class URDFVideoSimulator:
    """URDF Video Simulator"""
    
    def __init__(self, urdf_path: str, output_dir: Optional[str] = None):
        self.urdf_path = urdf_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(urdf_path), "simulation_output")
        self.cfg = self._create_default_config()
        
    def _create_default_config(self) -> DictConfig:
        """Create default configuration"""
        config = {
            "urdf": {
                "file": self.urdf_path,
                "raise_distance_file": None,
                "raise_distance_offset": 0.1,
                "rotation_pose": {
                    "rx": 0.0,
                    "ry": 0.0, 
                    "rz": 0.0
                }
            },
            "output": {
                "dir": self.output_dir,
                "seg_json": None
            },
            "simulation_params": {
                "joint_name": "all",
                "stationary_or_move": "move",
                "joint_move_dir": "auto",
                "num_steps": 50
            },
            "camera_params": {
                "width": 640,
                "height": 480,
                "views": {
                    "frontview": {
                        "cam_pos": [4, -3, 4],
                        "look_at": [0, 0, 0.6]
                    },
                    "sideview": {
                        "cam_pos": [-1, -4, 4],
                        "look_at": [0, 0, 0.8]
                    },
                    "topview": {
                        "cam_pos": [0, 0, 6],
                        "look_at": [0, 0, 0.8]
                    }
                }
            },
            "engine": {
                "timestep": 0.01
            },
            "lighting": {
                "ambient": [0.5, 0.5, 0.5],
                "directional": {
                    "direction": [0, 1, -1],
                    "color": [0.5, 0.5, 0.5]
                }
            },
            "floor_texture": "plain",
            "ray_tracing": False,
            "headless": True,
            "use_segmentation": False,
            "object_white": True,
            "flip_video": False
        }
        return OmegaConf.create(config)
    
    def set_camera_views(self, views: Dict[str, Dict[str, list]]):
        """Set camera views"""
        self.cfg.camera_params.views = views
    
    def set_simulation_params(self, joint_name: str = "all", num_steps: int = 60, 
                            move_direction: str = "auto"):
        """Set simulation parameters"""
        self.cfg.simulation_params.joint_name = joint_name
        self.cfg.simulation_params.num_steps = num_steps
        self.cfg.simulation_params.joint_move_dir = move_direction
    
    def set_video_options(self, use_segmentation: bool = False, object_white: bool = True,
                         flip_video: bool = False):
        """Set video options"""
        self.cfg.use_segmentation = use_segmentation
        self.cfg.object_white = object_white
        self.cfg.flip_video = flip_video
    
    def generate_video(self, mode: str = "move") -> bool:
        """
        Generate video
        
        Args:
            mode: "move" generates joint motion video, "stationary" generates static images
            
        Returns:
            Whether generation was successful
        """
        try:
            self.cfg.simulation_params.stationary_or_move = mode
            
            # Create output directory
            create_dir(self.cfg.output.dir)
            
            # Setup logging
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Starting to generate {mode} mode output")
            logging.info(f"URDF file: {self.cfg.urdf.file}")
            logging.info(f"Output directory: {self.cfg.output.dir}")
            
            # Initialize SAPIEN
            engine, scene, robot = setup_sapien(self.cfg)
            
            # Get manipulatable joints
            manipulatable_joints = get_manipulatable_joints(robot)
            logging.info(f"Found {len(manipulatable_joints)} manipulatable joints:")
            for joint_name, joint_type, limits in manipulatable_joints:
                logging.info(f"  - {joint_name} ({joint_type}): {limits}")
            
            # Setup cameras
            cameras = setup_cameras(self.cfg, scene)
            
            if mode == "move":
                capture_video(self.cfg, scene, robot, cameras, manipulatable_joints)
                logging.info("Video generation completed")
            elif mode == "stationary":
                capture_photo(self.cfg, scene, robot, cameras)
                logging.info("Image generation completed")
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            return True
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_joints(self) -> list:
        """List all manipulatable joints in the URDF file"""
        try:
            engine, scene, robot = setup_sapien(self.cfg)
            joints = get_manipulatable_joints(robot)
            return joints
        except Exception as e:
            logging.error(f"Failed to get joint information: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="URDF Video Simulator")
    parser.add_argument("--urdf", required=True, help="URDF file path")
    parser.add_argument("--output", help="Output directory path")
    parser.add_argument("--mode", choices=["move", "stationary"], default="move",
                       help="Generation mode: move=video, stationary=images")
    parser.add_argument("--joint", default="all", help="Joint name to simulate, default is all joints")
    parser.add_argument("--steps", type=int, default=60, help="Number of motion steps")
    parser.add_argument("--direction", choices=["auto", "move_up", "move_down"], 
                       default="auto", help="Joint motion direction")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation rendering")
    parser.add_argument("--flip", action="store_true", help="Flip video")
    parser.add_argument("--list-joints", action="store_true", help="List all manipulatable joints")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Check URDF file
    if not os.path.exists(args.urdf):
        print(f"Error: URDF file does not exist: {args.urdf}")
        sys.exit(1)
    
    # Create simulator
    simulator = URDFVideoSimulator(args.urdf, args.output)
    
    # Set parameters
    simulator.cfg.camera_params.width = args.width
    simulator.cfg.camera_params.height = args.height
    simulator.set_simulation_params(args.joint, args.steps, args.direction)
    simulator.set_video_options(args.segmentation, True, args.flip)
    
    # List joint information
    if args.list_joints:
        joints = simulator.list_joints()
        print("\nManipulatable joints:")
        for joint_name, joint_type, limits in joints:
            print(f"  - {joint_name} ({joint_type}): {limits}")
        return
    
    # Generate output
    print(f"Starting to generate {args.mode} mode output...")
    success = simulator.generate_video(args.mode)
    
    if success:
        print(f"Success! Output saved in: {simulator.output_dir}")
    else:
        print("Generation failed, please check error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()