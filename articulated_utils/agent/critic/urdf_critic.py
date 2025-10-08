"""
URDF Critic - Evaluation based on multi-view rendering
"""

import os
import logging
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod

try:
    import cv2
except ImportError:
    cv2 = None

from articulated_utils.utils.utils import join_path, create_dir, save_json


class BaseURDFCritic(ABC):
    """URDF Critic abstract base class"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate(self, urdf_path: str, iteration: int, seed: int) -> Dict[str, float]:
        """
        Evaluate URDF quality
        
        Args:
            urdf_path: URDF file path
            iteration: Current iteration number
            seed: Current seed
            
        Returns:
            Score dictionary containing individual scores and overall score
        """
        pass


class URDFRenderer:
    """URDF Renderer"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rendering configuration
        self.views = cfg.get('views', ['front', 'side', 'top', 'perspective'])
        self.resolution = cfg.get('resolution', [512, 512])
        self.use_sapien = cfg.get('use_sapien', True)
        
        # Camera view configuration - read from config file, use default values if not available
        self.camera_configs = cfg.get('camera_configs', {
            'front': {'pos': [2, 0, 1], 'target': [0, 0, 0.5]},
            'side': {'pos': [0, 2, 1], 'target': [0, 0, 0.5]},
            'top': {'pos': [0, 0, 3], 'target': [0, 0, 0]},
            'perspective': {'pos': [1.5, 1.5, 1.5], 'target': [0, 0, 0.5]}
        })
    
    def render_urdf(self, urdf_path: str, output_dir: str) -> Dict[str, str]:
        """
        Render URDF file to multi-view images
        
        Args:
            urdf_path: URDF file path
            output_dir: Output directory
            
        Returns:
            Mapping from view names to image paths
        """
        create_dir(output_dir)
        rendered_images = {}
        
        if not os.path.exists(urdf_path):
            self.logger.error(f"URDF file does not exist: {urdf_path}")
            return rendered_images
        
        try:
            if self.use_sapien:
                rendered_images = self._render_with_sapien(urdf_path, output_dir)
            else:
                rendered_images = self._render_with_pybullet(urdf_path, output_dir)
        except Exception as e:
            self.logger.error(f"Failed to render URDF: {e}")
        
        return rendered_images
    
    def _render_with_sapien(self, urdf_path: str, output_dir: str) -> Dict[str, str]:
        """Render using SAPIEN"""
        rendered_images = {}
        
        try:
            from articulated_utils.physics.sapien_render import SapienRenderer
            
            # Use existing SapienRenderer class
            renderer = SapienRenderer(self.resolution)
            
            for view_name in self.views:
                camera_config = self.camera_configs.get(view_name, self.camera_configs['front'])
                
                image_path = join_path(output_dir, f"{view_name}.png")
                
                # Call existing render_urdf method
                success = renderer.render_urdf(
                    urdf_path, 
                    image_path,
                    camera_pos=camera_config['pos'],
                    camera_target=camera_config['target']
                )
                
                if success:
                    rendered_images[view_name] = image_path
                    self.logger.info(f"Successfully rendered view {view_name}: {image_path}")
                else:
                    self.logger.warning(f"Failed to render view {view_name}")
            
            # Clean up renderer resources
            renderer.cleanup()
        
        except ImportError as e:
            self.logger.warning(f"SAPIEN unavailable: {e}, trying PyBullet")
            rendered_images = self._render_with_pybullet(urdf_path, output_dir)
        except Exception as e:
            self.logger.error(f"SAPIEN rendering failed: {e}")
            import traceback
            traceback.print_exc()
            # Try fallback to PyBullet
            self.logger.info("Trying fallback to PyBullet rendering")
            rendered_images = self._render_with_pybullet(urdf_path, output_dir)
        
        return rendered_images
    
    def _render_with_pybullet(self, urdf_path: str, output_dir: str) -> Dict[str, str]:
        """Render using PyBullet"""
        rendered_images = {}
        
        try:
            from articulated_utils.physics.pybullet_utils import PybulletRenderer
            
            # Use existing PybulletRenderer class
            renderer = PybulletRenderer(self.resolution)
            
            for view_name in self.views:
                camera_config = self.camera_configs.get(view_name, self.camera_configs['front'])
                
                image_path = join_path(output_dir, f"{view_name}.png")
                
                # Call existing render_urdf method
                success = renderer.render_urdf(
                    urdf_path,
                    image_path,
                    camera_pos=camera_config['pos'],
                    camera_target=camera_config['target']
                )
                
                if success:
                    rendered_images[view_name] = image_path
                    self.logger.info(f"Successfully rendered view {view_name}: {image_path}")
                else:
                    self.logger.warning(f"PyBullet failed to render view {view_name}")
            
            # Clean up renderer resources
            renderer.cleanup()
        
        except ImportError as e:
            self.logger.error(f"PyBullet unavailable: {e}")
        except Exception as e:
            self.logger.error(f"PyBullet rendering failed: {e}")
            import traceback
            traceback.print_exc()
        
        return rendered_images


class URDFScorer:
    """URDF Scorer"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Score weights
        self.position_weight = cfg.get('position_weight', 0.4)
        self.joint_weight = cfg.get('joint_weight', 0.6)
        
        # Scoring criteria
        self.min_parts_visible = cfg.get('min_parts_visible', 2)
        self.max_overlap_ratio = cfg.get('max_overlap_ratio', 0.3)
    
    def score_images(self, rendered_images: Dict[str, str], urdf_path: str) -> Dict[str, float]:
        """Calculate scores based on rendered images (10-point scale) - evaluate only position and joints"""
        scores = {
            'position_score': 0.0,
            'joint_score': 0.0,
            'overall_score': 0.0
        }
        
        if not rendered_images:
            self.logger.warning("No rendered images, returning zero score")
            return scores
        
        try:
            # Evaluate position reasonableness (0-10 points, integer)
            position_score = round(self._evaluate_positions(rendered_images) * 10)
            scores['position_score'] = position_score
            
            # Evaluate joint reasonableness (0-10 points, integer) - including joint position and joint type
            joint_score = round(self._evaluate_joints(rendered_images, urdf_path) * 10)
            scores['joint_score'] = joint_score
            
            # Calculate total score (0-10 points, integer) - only consider position and joints
            overall_score = round(
                self.position_weight * position_score + 
                self.joint_weight * joint_score
            )
            scores['overall_score'] = overall_score
            
            self.logger.info(f"Scoring result (10-point integer scale): position={position_score}/10, joint={joint_score}/10, "
                           f"total={overall_score}/10")
        
        except Exception as e:
            self.logger.error(f"Image scoring failed: {e}")
        
        return scores
    
    def _evaluate_visibility(self, rendered_images: Dict[str, str]) -> float:
        """Evaluate component visibility"""
        if not cv2:
            self.logger.warning("OpenCV unavailable, skipping visibility evaluation")
            return 0.5  # Default score
        
        visible_parts = 0
        total_views = len(rendered_images)
        
        for view_name, image_path in rendered_images.items():
            if os.path.exists(image_path):
                try:
                    # 读取图像并分析
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Simple visibility detection: check non-background areas in image
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        non_bg_pixels = np.sum(gray > 50)  # Assume background is black or dark
                        total_pixels = gray.shape[0] * gray.shape[1]
                        
                        if non_bg_pixels / total_pixels > 0.1:  # More than 10% pixels are non-background
                            visible_parts += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze image {image_path}: {e}")
        
        visibility_score = visible_parts / max(total_views, 1)
        return min(visibility_score, 1.0)
    
    def _evaluate_positions(self, rendered_images: Dict[str, str]) -> float:
        """Evaluate component position reasonableness - based on multi-view analysis"""
        if not cv2:
            return 0.5
        
        position_scores = []
        
        for view_name, image_path in rendered_images.items():
            if os.path.exists(image_path):
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Analyze image center of mass distribution
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate center of mass offset (ideally should be near image center)
                        moments = cv2.moments(gray)
                        if moments['m00'] > 0:
                            cx = moments['m10'] / moments['m00']
                            cy = moments['m01'] / moments['m00']
                            
                            center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
                            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                            max_distance = np.sqrt(center_x**2 + center_y**2)
                            
                            # Closer distance gets higher score
                            center_score = 1.0 - min(distance / max_distance, 1.0)
                            
                            # Analyze image coverage range (avoid parts being too small or too large)
                            non_zero_pixels = np.sum(gray > 50)
                            total_pixels = gray.shape[0] * gray.shape[1]
                            coverage_ratio = non_zero_pixels / total_pixels
                            
                            # Ideal coverage range is 10%-80%
                            if 0.1 <= coverage_ratio <= 0.8:
                                coverage_score = 1.0
                            elif coverage_ratio < 0.1:
                                coverage_score = coverage_ratio / 0.1
                            else:
                                coverage_score = max(0.0, 1.0 - (coverage_ratio - 0.8) / 0.2)
                            
                            # Comprehensive score
                            view_score = (center_score * 0.6 + coverage_score * 0.4)
                            position_scores.append(view_score)
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze position {image_path}: {e}")
        
        return np.mean(position_scores) if position_scores else 0.5
    
    def _evaluate_joints(self, rendered_images: Dict[str, str], urdf_path: str) -> float:
        """Evaluate joint reasonableness - including joint position and joint type"""
        try:
            # Read URDF file and analyze joints
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            
            # Analyze joint count and types
            joint_count = urdf_content.count('<joint')
            link_count = urdf_content.count('<link')
            
            # Analyze joint types
            fixed_joints = urdf_content.count('type="fixed"')
            revolute_joints = urdf_content.count('type="revolute"')
            prismatic_joints = urdf_content.count('type="prismatic"')
            continuous_joints = urdf_content.count('type="continuous"')
            
            # Analyze joint positions (through origin tags)
            origin_count = urdf_content.count('<origin')
            axis_count = urdf_content.count('<axis')
            
            # Calculate joint type score - judge if type is correct based on common sense
            type_score = 0.0
            if joint_count > 0:
                # Analyze URDF content, judge if joint types are reasonable
                urdf_lower = urdf_content.lower()
                
                # Judge appropriate joint types based on object type
                if 'sliding' in urdf_lower or 'lid' in urdf_lower or 'drawer' in urdf_lower:
                    # Sliding lids and drawers should use prismatic joints
                    if prismatic_joints > 0:
                        type_score += 0.5
                        self.logger.info("Detected sliding component, using prismatic joint - correct")
                    else:
                        type_score += 0.1
                        self.logger.warning("Detected sliding component, but not using prismatic joint - possibly incorrect")
                
                if 'hinge' in urdf_lower or 'door' in urdf_lower or 'handle' in urdf_lower:
                    # Hinges, doors, handles should use revolute joints
                    if revolute_joints > 0:
                        type_score += 0.5
                        self.logger.info("Detected rotating component, using revolute joint - correct")
                    else:
                        type_score += 0.1
                        self.logger.warning("Detected rotating component, but not using revolute joint - possibly incorrect")
                
                if 'wheel' in urdf_lower or 'continuous' in urdf_lower:
                    # Wheels should use continuous joints
                    if continuous_joints > 0:
                        type_score += 0.5
                        self.logger.info("Detected continuous rotation component, using continuous joint - correct")
                    else:
                        type_score += 0.1
                        self.logger.warning("Detected continuous rotation component, but not using continuous joint - possibly incorrect")
                
                # Using fixed joints for basic connections is reasonable
                if fixed_joints > 0:
                    type_score += 0.2
                    self.logger.info(f"Using {fixed_joints} fixed joints for basic connections - reasonable")
                
                # Avoid too many fixed joints (over 80%)
                if fixed_joints > joint_count * 0.8:
                    type_score *= 0.3
                    self.logger.warning(f"Too many fixed joints ({fixed_joints}/{joint_count}), may lack mobility")
                
                # If no motion joints, score lower
                if prismatic_joints == 0 and revolute_joints == 0 and continuous_joints == 0:
                    type_score *= 0.2
                    self.logger.warning("No motion joints, object may be unable to move")
            
            # Calculate joint position score
            position_score = 0.0
            if origin_count > 0:
                # Has explicit joint position definition
                position_score += 0.3
            if axis_count > 0:
                # Has explicit joint axis definition
                position_score += 0.3
            
            # Calculate joint quantity reasonableness score
            quantity_score = 0.0
            if link_count > 1:
                expected_joint_ratio = (link_count - 1) / link_count
                actual_joint_ratio = joint_count / link_count if link_count > 0 else 0
                
                # Calculate score based on ratio
                ratio_score = 1.0 - abs(actual_joint_ratio - expected_joint_ratio)
                quantity_score = max(ratio_score, 0.0) * 0.4
            else:
                quantity_score = 0.2  # Default score for single link
            
            # **New: Joint axis alignment reasonableness evaluation**
            axis_score = self._evaluate_joint_axis_alignment(urdf_content)
            
            # **New: Joint position geometric reasonableness evaluation**
            geometry_score = self._evaluate_joint_geometry(urdf_content)
            
            # Comprehensive score, adding new axis and geometric evaluations
            total_score = type_score * 0.4 + position_score * 0.2 + quantity_score * 0.1 + axis_score * 0.2 + geometry_score * 0.1
            return min(total_score, 1.0)  # Ensure not exceeding 1.0
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze joints: {e}")
            return 0.5
    
    def _evaluate_joint_axis_alignment(self, urdf_content: str) -> float:
        """Evaluate joint axis alignment reasonableness (general method)"""
        import re
        
        axis_score = 0.0
        issues = []
        
        try:
            # Extract all joint information
            joint_pattern = r'<joint[^>]*name="([^"]*)"[^>]*>(.*?)</joint>'
            joints = re.findall(joint_pattern, urdf_content, re.DOTALL)
            
            for joint_name, joint_content in joints:
                # Extract joint type
                type_match = re.search(r'type="([^"]*)"', joint_content)
                if not type_match:
                    continue
                    
                joint_type = type_match.group(1)
                
                # Extract joint axis
                axis_match = re.search(r'<axis[^>]*xyz="([^"]*)"', joint_content)
                if axis_match:
                    axis_xyz = axis_match.group(1).split()
                    if len(axis_xyz) == 3:
                        try:
                            axis = [float(x) for x in axis_xyz]
                            
                            # General axis normalization check
                            if joint_type in ['revolute', 'continuous', 'prismatic']:
                                axis_magnitude = np.sqrt(sum(x**2 for x in axis))
                                
                                if abs(axis_magnitude - 1.0) > 0.1:
                                    issues.append(f"Joint {joint_name} axis not normalized (magnitude={axis_magnitude:.3f})")
                                    axis_score -= 0.1
                                else:
                                    axis_score += 0.1
                                    
                        except ValueError:
                            issues.append(f"Joint {joint_name} axis numerical format error")
                            axis_score -= 0.1
                else:
                    if joint_type in ['revolute', 'continuous', 'prismatic']:
                        issues.append(f"Motion joint {joint_name} missing axis definition")
                        axis_score -= 0.15
                
                # General joint position reasonableness check
                origin_match = re.search(r'<origin[^>]*xyz="([^"]*)"', joint_content)
                if origin_match:
                    origin_xyz = origin_match.group(1).split()
                    if len(origin_xyz) == 3:
                        try:
                            origin = [float(x) for x in origin_xyz]
                            
                            # Check for abnormally distant joint origins (general standard)
                            distance_from_origin = np.sqrt(sum(x**2 for x in origin))
                            if distance_from_origin > 10.0:  # General threshold: over 10 units away
                                issues.append(f"🚨 Joint {joint_name} position too far (distance from origin={distance_from_origin:.2f})")
                                axis_score -= 0.3  # Serious problem
                            elif distance_from_origin > 5.0:  # General threshold: over 5 units
                                issues.append(f"⚠️  Joint {joint_name} position relatively far (distance from origin={distance_from_origin:.2f})")
                                axis_score -= 0.15
                                
                        except ValueError:
                            issues.append(f"Joint {joint_name} position numerical format error")
                            axis_score -= 0.1
            
            # Record discovered problems
            if issues:
                self.logger.warning("🔍 Joint axis analysis found problems:")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("✅ Joint axis check passed")
            
            # Normalize score to 0-1 range
            return max(0.0, min(1.0, axis_score + 0.5))  # Base score 0.5
            
        except Exception as e:
            self.logger.error(f"Joint axis evaluation failed: {e}")
            return 0.3
    
    def _evaluate_joint_geometry(self, urdf_content: str) -> float:
        """Evaluate joint geometric reasonableness (general method)"""
        import re
        
        geometry_score = 0.5  # Base score
        issues = []
        
        try:
            # Extract all joint information
            joint_pattern = r'<joint[^>]*name="([^"]*)"[^>]*>(.*?)</joint>'
            joints = re.findall(joint_pattern, urdf_content, re.DOTALL)
            
            # Build joint connection relationship graph
            joint_connections = {}
            joint_types = {}
            joint_axes = {}
            
            for joint_name, joint_content in joints:
                # Extract parent-child relationships
                parent_match = re.search(r'<parent[^>]*link="([^"]*)"', joint_content)
                child_match = re.search(r'<child[^>]*link="([^"]*)"', joint_content)
                
                if parent_match and child_match:
                    parent_link = parent_match.group(1)
                    child_link = child_match.group(1)
                    joint_connections[joint_name] = {
                        'parent': parent_link,
                        'child': child_link
                    }
                
                # Extract joint type
                type_match = re.search(r'type="([^"]*)"', joint_content)
                if type_match:
                    joint_types[joint_name] = type_match.group(1)
                
                # Extract joint axis
                axis_match = re.search(r'<axis[^>]*xyz="([^"]*)"', joint_content)
                if axis_match:
                    axis_xyz = axis_match.group(1).split()
                    if len(axis_xyz) == 3:
                        try:
                            joint_axes[joint_name] = [float(x) for x in axis_xyz]
                        except ValueError:
                            pass
            
            # General geometric reasonableness check
            for joint_name in joint_connections.keys():
                joint_type = joint_types.get(joint_name)
                joint_axis = joint_axes.get(joint_name)
                
                # Check semantic consistency between joint type and name (not dependent on specific objects)
                if joint_type and joint_axis:
                    # General check: continuous rotation joint axis should be normalized
                    if joint_type == 'continuous':
                        axis_magnitude = np.sqrt(sum(x**2 for x in joint_axis))
                        if abs(axis_magnitude - 1.0) > 0.1:
                            issues.append(f"Continuous rotation joint {joint_name} axis not normalized")
                            geometry_score -= 0.1
                        else:
                            geometry_score += 0.05
                    
                    # General check: prismatic joint axis should be normalized
                    elif joint_type == 'prismatic':
                        axis_magnitude = np.sqrt(sum(x**2 for x in joint_axis))
                        if abs(axis_magnitude - 1.0) > 0.1:
                            issues.append(f"Prismatic joint {joint_name} axis not normalized")
                            geometry_score -= 0.1
                        else:
                            geometry_score += 0.05
                    
                    # General check: revolute joint axis should be normalized
                    elif joint_type == 'revolute':
                        axis_magnitude = np.sqrt(sum(x**2 for x in joint_axis))
                        if abs(axis_magnitude - 1.0) > 0.1:
                            issues.append(f"Revolute joint {joint_name} axis not normalized")
                            geometry_score -= 0.1
                        else:
                            geometry_score += 0.05
            
            # General check: check if there is reasonable distribution of motion joints
            total_joints = len(joints)
            movable_joints = sum(1 for jtype in joint_types.values() 
                               if jtype in ['revolute', 'continuous', 'prismatic'])
            
            if total_joints > 1:
                movable_ratio = movable_joints / total_joints
                if movable_ratio > 0.1:  # At least 10% of joints are movable
                    geometry_score += 0.1
                else:
                    issues.append(f"Movable joint ratio too low ({movable_joints}/{total_joints})")
                    geometry_score -= 0.1
            
            # Record geometric problems
            if issues:
                self.logger.warning("🔧 Joint geometric analysis found problems:")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("✅ Joint geometric check passed")
            
            return max(0.0, min(1.0, geometry_score))
            
        except Exception as e:
            self.logger.error(f"Joint geometric evaluation failed: {e}")
            return 0.3


class URDFCritic(BaseURDFCritic):
    """Complete URDF critic implementation"""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        
        # Initialize renderer and scorer
        renderer_cfg = cfg.get('renderer', {})
        scorer_cfg = cfg.get('scorer', {})
        
        self.renderer = URDFRenderer(renderer_cfg)
        self.scorer = URDFScorer(scorer_cfg)
        
        self.logger.info("URDFCritic initialization completed")
    
    def evaluate(self, urdf_path: str, iteration: int, seed: int) -> Dict[str, float]:
        """Evaluate URDF quality"""
        self.logger.info(f"Evaluating URDF: {urdf_path} (iteration{iteration}, seed{seed})")
        
        # Create output directory
        output_dir = join_path(
            os.path.dirname(urdf_path),
            f"critic_renders_iter{iteration}_seed{seed}"
        )
        
        # Render URDF
        rendered_images = self.renderer.render_urdf(urdf_path, output_dir)
        
        # Score
        scores = self.scorer.score_images(rendered_images, urdf_path)
        
        # Add metadata
        scores.update({
            'iteration': iteration,
            'seed': seed,
            'urdf_path': urdf_path,
            'rendered_images': rendered_images,
            'feedback_score': scores['overall_score']  # Compatible with actor-critic framework
        })
        
        return scores


class PromptCritic(BaseURDFCritic):
    """Prompt-based URDF critic - 3D model plausibility and functionality expert"""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.renderer = URDFRenderer(cfg.get('renderer', {}))
        # Don't create output directory here, create in evaluate method based on URDF path
        
        # Prompt template
        self.prompt_template = """## ROLE: PromptCritic - 3D Model Plausibility & Functionality Expert

You are a `PromptCritic`, an AI expert performing a comprehensive evaluation of a 3D model. You will assess link positions using only visual renders, and then perform a fused vision-and-data evaluation of the joints. Your evaluation must be based on general world knowledge of how objects are constructed and function.

## INPUTS:
1.  **Multi-View Renders**: A set of rendered images of the predicted 3D model.
2.  **URDF File Content**: The full XML content of the URDF file defining the model's links and joints.

## TASK:
Perform two separate evaluations as detailed below and provide the output in a single JSON object.

### Task 1: Evaluate Link Position (Using Images ONLY)
- Based **exclusively on the Multi-View Rendered Images**, evaluate the physical plausibility of the link positions.
- Check for intersections, floating parts, and incorrect alignment or symmetry.
- Provide a single score reflecting the overall positional quality.

### Task 2: Evaluate Joints (Using URDF AND Images)
- Based on a **combined analysis of the URDF File Content AND the Multi-View Rendered Images**, evaluate each joint.
- For each joint, evaluate two parameters:
    1.  **Type**: Analyze the `type` attribute in the URDF. Use the visual context from the images to confirm if this type is functionally correct for the connected parts.
    2.  **Position**: Analyze the `<origin>` tag in the URDF. Visually locate this coordinate on the rendered images and determine if it is a logical pivot or connection point for the parts involved.
- In your reasoning, you must refer to both the URDF values and visual evidence from the images.

## OUTPUT FORMAT (Strict JSON):
Return your complete evaluation in the following JSON format. Set the final `success` flag to `true` only if both the `link_position_score` and `joint_score` are 8.0 or higher.

```json
{{
  "link_position_score": <float, 0-10>,
  "link_position_reasoning": "<string, describe visual issues like intersections or floating parts.>",
  "joint_score": <float, 0-10, the final average score for all joints>,
  "joint_evaluation_details": [
    {{
      "joint_name": "<string, name of the joint from URDF>",
      "type_score": <float, 0-10>,
      "position_score": <float, 0-10>,
      "reasoning": "<string, explain reasoning by referencing both URDF (`origin xyz=...`) and visual evidence (`...which visually corresponds to the center of the drawer, an incorrect location for a slide joint.`).>"
    }}
  ],
  "success": <boolean, true or false>
}}
```

## URDF CONTENT:
{urdf_content}

## RENDERED IMAGES:
{image_descriptions}

Please provide your evaluation in the exact JSON format specified above."""
        
        self.logger.info("PromptCritic initialization completed")
    
    def evaluate(self, urdf_path: str, iteration: int, seed: int) -> Dict[str, Any]:
        """
        评估URDF质量 - 使用PromptCritic方法
        
        Args:
            urdf_path: URDF文件路径
            iteration: 当前迭代次数
            seed: 当前种子
            
        Returns:
            评分字典，包含各项评分、详细分析和success标志
        """
        self.logger.info(f"PromptCritic评估URDF: {urdf_path} (迭代{iteration}, 种子{seed})")
        
        # Create output directory
        output_dir = join_path(
            os.path.dirname(urdf_path),
            f"critic_renders_iter{iteration}_seed{seed}"
        )
        
        # 渲染多视角图像
        rendered_images = self.renderer.render_urdf(urdf_path, output_dir)
        
        # 读取URDF内容
        try:
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
        except Exception as e:
            self.logger.error(f"读取URDF文件失败: {e}")
            return self._create_fallback_result()
        
        # 准备图像描述
        image_descriptions = self._prepare_image_descriptions(rendered_images)
        
        # 构建完整的prompt
        full_prompt = self.prompt_template.format(
            urdf_content=urdf_content,
            image_descriptions=image_descriptions
        )
        
        # 调用LLM进行评估（这里需要集成实际的LLM调用）
        evaluation_result = self._call_llm_for_evaluation(full_prompt)
        
        # 保存结果到文件 - 保存到URDF文件所在目录
        urdf_dir = os.path.dirname(urdf_path)
        result_file = join_path(urdf_dir, f"prompt_critic_result_iter{iteration}_seed{seed}.json")
        save_json(evaluation_result, result_file)
        self.logger.info(f"PromptCritic评估结果已保存到: {result_file}")
        
        # 转换为标准格式
        standard_scores = self._convert_to_standard_format(evaluation_result)
        
        # Add metadata
        standard_scores.update({
            'iteration': iteration,
            'seed': seed,
            'urdf_path': urdf_path,
            'rendered_images': rendered_images,
            'feedback_score': standard_scores['overall_score']  # 兼容actor-critic框架
        })
        
        return standard_scores
    
    def _prepare_image_descriptions(self, rendered_images: Dict[str, str]) -> str:
        """准备图像描述"""
        descriptions = []
        for view_name, image_path in rendered_images.items():
            if os.path.exists(image_path):
                descriptions.append(f"- {view_name}: {image_path}")
            else:
                descriptions.append(f"- {view_name}: [Image not found]")
        return "\n".join(descriptions)
    
    def _call_llm_for_evaluation(self, prompt: str) -> Dict[str, Any]:
        """
        调用LLM进行评估
        注意：这里需要集成实际的LLM API调用
        目前返回模拟结果
        """
        # TODO: 集成实际的LLM API调用
        # 这里返回一个模拟的评估结果
        mock_result = {
            "link_position_score": 8.5,
            "link_position_reasoning": "The rendered images show proper alignment of the pot body and sliding lid. No intersections or floating parts detected. The lid appears to be correctly positioned above the pot body.",
            "joint_score": 9.0,
            "joint_evaluation_details": [
                {
                    "joint_name": "base_to_kitchen_pot_body",
                    "type_score": 10.0,
                    "position_score": 10.0,
                    "reasoning": "Fixed joint type is appropriate for connecting the base to the pot body. The origin coordinates in the URDF correspond to the bottom center of the pot, which is visually correct for a stable base connection."
                },
                {
                    "joint_name": "kitchen_pot_body_to_sliding_lid",
                    "type_score": 9.0,
                    "position_score": 8.0,
                    "reasoning": "Prismatic joint type is functionally correct for a sliding lid mechanism. The origin coordinates in the URDF (xyz=\"0 0 0.5\") visually correspond to the top center of the pot body, which is a logical connection point for the sliding lid."
                }
            ],
            "success": True
        }
        
        self.logger.info("Using mock LLM evaluation result (need to integrate actual LLM API)")
        return mock_result
    
    def _convert_to_standard_format(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PromptCritic results to standard format"""
        # Calculate average joint scores
        joint_details = evaluation_result.get('joint_evaluation_details', [])
        if joint_details:
            avg_type_score = np.mean([j.get('type_score', 0) for j in joint_details])
            avg_position_score = np.mean([j.get('position_score', 0) for j in joint_details])
            joint_score = (avg_type_score + avg_position_score) / 2
        else:
            joint_score = evaluation_result.get('joint_score', 0)
        
        # Convert to standard format
        standard_result = {
            'position_score': evaluation_result.get('link_position_score', 0),
            'joint_score': joint_score,
            'overall_score': (evaluation_result.get('link_position_score', 0) + joint_score) / 2,
            'success': evaluation_result.get('success', False),
            'detailed_analysis': evaluation_result,
            'link_position_reasoning': evaluation_result.get('link_position_reasoning', ''),
            'joint_evaluation_details': joint_details
        }
        
        self.logger.info(f"PromptCritic scoring result: position={standard_result['position_score']}/10, "
                        f"joint={standard_result['joint_score']}/10, "
                        f"total={standard_result['overall_score']}/10, "
                        f"success={standard_result['success']}")
        
        return standard_result
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result"""
        return {
            'position_score': 0,
            'joint_score': 0,
            'overall_score': 0,
            'success': False,
            'detailed_analysis': {},
            'link_position_reasoning': 'Failed to read URDF file',
            'joint_evaluation_details': [],
            'iteration': 0,
            'seed': 0,
            'urdf_path': '',
            'rendered_images': {},
            'feedback_score': 0
        }