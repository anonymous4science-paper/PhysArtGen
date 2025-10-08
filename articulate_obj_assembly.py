"""
OBJ file-based assembly system with integrated Critic-Verifier architecture
Supports multi-part assembly and iterative optimization while maintaining relative positional relationships
"""

import datetime
import hydra
import logging
import json
import os
import shutil
from typing import Dict, List, Optional, Callable, Any
import time
import xml.etree.ElementTree as ET
import numpy as np
from omegaconf import DictConfig, OmegaConf

from articulated_utils.utils.utils import seed_everything, Steps, save_json, join_path, create_dir
from articulate_link import articulate_link
from articulate_joint import articulate_joint
from articulated_utils.agent.critic.urdf_critic import URDFCritic, PromptCritic
import traceback

logging.basicConfig(level=logging.INFO)


class OBJProcessor:
    """Process OBJ files while maintaining relative positional relationships"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.obj_info = {}
    
    def validate_obj_files(self, obj_paths: Dict[str, str]) -> bool:
        """Validate if OBJ files exist and are valid"""
        if len(obj_paths) > 6:
            logging.warning(f"Provided {len(obj_paths)} OBJ files, exceeding the recommended limit of 6")
        
        for name, path in obj_paths.items():
            if not os.path.exists(path):
                logging.error(f"OBJ file does not exist: {name} -> {path}")
                return False
            
            # Simple OBJ file format validation
            try:
                with open(path, 'r') as f:
                    first_lines = f.readlines()[:10]
                    if not any(line.startswith('v ') for line in first_lines):
                        logging.warning(f"OBJ file may be invalid (no vertex data found): {path}")
            except Exception as e:
                logging.error(f"Failed to read OBJ file: {path}, error: {e}")
                return False
        
        return True
    
    def extract_obj_metadata(self, obj_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Extract OBJ file metadata including bounding box and center point"""
        metadata = {}
        
        for name, path in obj_paths.items():
            try:
                vertices = []
                with open(path, 'r') as f:
                    for line in f:
                        if line.startswith('v '):
                            coords = line.strip().split()[1:4]
                            vertices.append([float(x) for x in coords])
                
                if vertices:
                    import numpy as np
                    vertices = np.array(vertices)
                    bbox_min = vertices.min(axis=0)
                    bbox_max = vertices.max(axis=0)
                    center = (bbox_min + bbox_max) / 2
                    size = bbox_max - bbox_min
                    
                    metadata[name] = {
                        'bbox_min': bbox_min.tolist(),
                        'bbox_max': bbox_max.tolist(),
                        'center': center.tolist(),
                        'size': size.tolist(),
                        'vertex_count': len(vertices)
                    }
                    logging.info(f"OBJ metadata {name}: center={center}, size={size}")
                else:
                    logging.warning(f"No vertex data found: {path}")
                    
            except Exception as e:
                logging.error(f"Failed to extract OBJ metadata: {path}, error: {e}")
        
        return metadata


def create_mock_mesh_retrieval_steps(cfg: DictConfig, obj_paths: Dict[str, str]) -> Steps:
    """Create mock mesh retrieval steps while maintaining compatibility with existing system"""
    steps = Steps()
    
    # Create necessary directories
    create_dir(join_path(cfg.out_dir, "task_specifier"))
    create_dir(join_path(cfg.out_dir, "box_layout"))
    create_dir(join_path(cfg.out_dir, "mesh_retrieval"))
    
    # 1. Task specifier: Generate task description based on prompt
    from articulated_utils.agent.actor.mesh_retrieval.text_task_specifier import TextTaskSpecifier
    from articulated_utils.utils.utils import create_task_config
    
    task_specifier = TextTaskSpecifier(create_task_config(cfg, "task_specifier"))
    task_specifier.generate_prediction(cfg.prompt, **cfg.gen_config)
    steps.add_step("Task Specification", task_specifier)

    # 2. Layout planner: Generate component layout
    from articulated_utils.agent.actor.mesh_retrieval.text_layout_planner import TextLayoutPlanner
    
    layout_planner = TextLayoutPlanner(create_task_config(cfg, "box_layout"))
    
    # Build prompt containing user OBJ names
    task_prediction = task_specifier.load_prediction()
    print(f"Debug: task_prediction = {task_prediction}")
    if task_prediction is None:
        raise ValueError("Task prediction is None")
    if 'output' not in task_prediction:
        raise ValueError(f"Task prediction missing 'output' key. Keys: {task_prediction.keys()}")
    base_prompt = task_prediction['output']
    obj_names = list(obj_paths.keys())
    enhanced_prompt = f"{base_prompt} you must Use these exact component names: {', '.join(obj_names)}."
    
    layout_planner.generate_prediction(enhanced_prompt, **cfg.gen_config)
    steps.add_step("Box Layout", layout_planner)

    # 3. Create OBJ-based mesh retrieval results
    class OBJMeshRetriever:
        def __init__(self, cfg, obj_paths, obj_processor, layout_planner):
            self.cfg = cfg
            self.obj_paths = obj_paths
            self.obj_processor = obj_processor
            self.layout_planner = layout_planner
            self.out_dir = join_path(cfg.out_dir, "mesh_retrieval")
            create_dir(self.out_dir)
        
        def load_prediction(self):
            meshes = {}
            
            # Get layout generated by TextLayoutPlanner
            box_layout = self.layout_planner.load_prediction()
            obj_paths_list = list(self.obj_paths.items())
            
            # Directly match user-provided OBJ files by name
            for box in box_layout:
                layout_name = box["name"]
                
                # Find matching user OBJ file
                if layout_name in self.obj_paths:
                    path = self.obj_paths[layout_name]
                    meshes[layout_name] = {
                        'mesh_file': path,
                        'cosine_similarity': 0.95,  # High similarity since it's user-provided exact match
                        'mesh_description': f"User-provided {layout_name} component",
                        'link_description': f"{layout_name}. User-provided component for the assembly"
                    }
                    logging.info(f"Direct name match: {layout_name} -> {path}")
                else:
                    # If no matching user OBJ file found, use placeholder
                    meshes[layout_name] = {
                        'mesh_file': f"placeholder_{layout_name}.obj",
                        'cosine_similarity': 0.5,
                        'mesh_description': f"Placeholder for {layout_name}",
                        'link_description': f"{layout_name}. Placeholder component"
                    }
                    logging.warning(f"No matching OBJ file found, using placeholder: {layout_name}")
            
            # Save results
            save_json(meshes, join_path(self.out_dir, "semantic_search_result.json"))
            save_json(meshes, join_path(self.out_dir, "semantic_search_all_result.json"))
            
            return meshes

    # Create OBJ processor and add to steps
    obj_processor = OBJProcessor(cfg)
    mesh_searcher = OBJMeshRetriever(cfg, obj_paths, obj_processor, layout_planner)
    steps.add_step("Mesh Retrieval", mesh_searcher)

    return steps


class OBJAssemblyPipeline:
    """OBJ file-based assembly pipeline"""
    
    def __init__(self, cfg: DictConfig, obj_paths: Dict[str, str]):
        self.cfg = cfg
        self.obj_paths = obj_paths
        self.steps = Steps()
        self.history = []  # Store historical iteration results
        self.feedback_history = []  # Store structured feedback history
        
        # Validate input
        obj_processor = OBJProcessor(cfg)
        if not obj_processor.validate_obj_files(obj_paths):
            raise ValueError("OBJ file validation failed")
        
        # Extract metadata
        self.obj_metadata = obj_processor.extract_obj_metadata(obj_paths)
        
        # Set random seed
        seed_everything(cfg.get('seed', 0))
        
        # Create output directory
        create_dir(cfg.out_dir)
        
        logging.info(f"Initialize OBJ assembly pipeline: {len(obj_paths)} components")
        logging.info(f"Component list: {list(obj_paths.keys())}")
    
    def run_single_iteration(self, iteration: int, seed: int, feedback: Dict = None, retry_kwargs: Dict = None) -> Dict:
        """Run single assembly iteration with historical feedback support"""
        logging.info(f"Starting iteration {iteration}")
        
        try:
            # 1. Adjust prompt based on historical feedback
            enhanced_prompt = self._enhance_prompt_with_feedback(self.cfg.prompt, feedback)
            
            # 2. Create independent configuration copy for each iteration
            iter_cfg = self.cfg.copy()
            iter_cfg.gen_config.overwrite = True  # Force regeneration
            
            # 3. Prepare retry_kwargs containing Critic feedback
            if retry_kwargs is None:
                retry_kwargs = {}
            
            # Add feedback information to retry_kwargs
            if feedback:
                retry_kwargs.update({
                    'previous_feedback': feedback,
                    'improvement_suggestions': self._extract_improvement_suggestions(feedback),
                    'failure_analysis': self._extract_failure_analysis(feedback)
                })
                logging.info(f"Passing retry_kwargs to Actor: {list(retry_kwargs.keys())}")
            
            # 4. Create new Steps object to avoid reuse
            current_steps = Steps()
            
            # 5. Create mesh retrieval steps
            mesh_steps = create_mock_mesh_retrieval_steps(iter_cfg, self.obj_paths)
            
            # Merge steps
            for step_name, step_data in mesh_steps:
                current_steps.add_step(step_name, step_data)
            
            # 6. Link assembly (pass retry_kwargs)
            logging.info(f"Executing Link assembly... (enhanced prompt: {enhanced_prompt[:100]}...)")
            link_result = self._run_link_with_retry_kwargs(
                enhanced_prompt, current_steps, str(iter_cfg.gpu_id), iter_cfg, retry_kwargs
            )
            
            # 7. Joint assembly (pass retry_kwargs)
            logging.info("Executing Joint assembly...")
            joint_result = self._run_joint_with_retry_kwargs(
                enhanced_prompt, current_steps, str(iter_cfg.gpu_id), iter_cfg, retry_kwargs
            )
            
            # Update main steps object
            self.steps = current_steps
            
            # Return results
            result = {
                'iteration': iteration,
                'seed': seed,
                'link_result': link_result,
                'joint_result': joint_result,
                'enhanced_prompt': enhanced_prompt,
                'feedback_used': feedback is not None,
                'success': True
            }
            
            logging.info(f"Iteration {iteration}-{seed} completed")
            return result
            
        except Exception as e:
            logging.error(f"Iteration {iteration}-{seed} failed: {e}")
            return {
                'iteration': iteration,
                'seed': seed,
                'success': False,
                'error': str(e)
            }
    
    def _enhance_prompt_with_feedback(self, base_prompt: str, feedback: Dict) -> str:
        """Enhance prompt based on historical feedback"""
        if not feedback:
            return base_prompt
        
        enhancement = ""
        
        # Add feedback based on position score
        pos_score = feedback.get('position_score', 0)
        if pos_score < 5.0:
            enhancement += " Pay special attention to component positioning and spatial relationships."
        
        # Add feedback based on joint score
        joint_score = feedback.get('joint_score', 0)
        if joint_score < 5.0:
            enhancement += " Focus on joint axis alignment and parameter accuracy."
        
        # Add guidance based on specific feedback text
        if 'detailed_feedback' in feedback:
            enhancement += f" Previous iteration feedback: {feedback['detailed_feedback']}"
        
        # Add guidance based on historical errors
        if len(self.feedback_history) > 0:
            common_issues = self._analyze_common_issues()
            if common_issues:
                enhancement += f" Avoid these recurring issues: {common_issues}"
        
        return base_prompt + enhancement
    
    def _analyze_common_issues(self) -> str:
        """Analyze common issues in historical feedback"""
        if not self.feedback_history:
            return ""
        
        issues = []
        pos_scores = [f.get('position_score', 0) for f in self.feedback_history]
        joint_scores = [f.get('joint_score', 0) for f in self.feedback_history]
        
        if sum(pos_scores) / len(pos_scores) < 5.0:
            issues.append("positioning accuracy")
        
        if sum(joint_scores) / len(joint_scores) < 5.0:
            issues.append("joint parameter settings")
        
        return ", ".join(issues)
    
    def _extract_improvement_suggestions(self, feedback: Dict) -> str:
        """Extract improvement suggestions from feedback"""
        suggestions = []
        
        # Extract suggestions based on different types of feedback
        if 'detailed_feedback' in feedback:
            suggestions.append(feedback['detailed_feedback'])
        
        # Give specific suggestions based on scores
        pos_score = feedback.get('position_score', 10)
        joint_score = feedback.get('joint_score', 10)
        
        if pos_score < 5.0:
            suggestions.append("Improve component positioning and spatial relationships")
        if joint_score < 5.0:
            suggestions.append("Improve joint axis alignment and parameter accuracy")
            
        # If using PromptCritic, may have more detailed improvement suggestions
        if 'improvement_suggestion' in feedback:
            suggestions.append(feedback['improvement_suggestion'])
            
        return " | ".join(suggestions) if suggestions else ""
    
    def _extract_failure_analysis(self, feedback: Dict) -> Dict:
        """Extract failure analysis from feedback"""
        analysis = {}
        
        # Position issue analysis
        pos_score = feedback.get('position_score', 10)
        if pos_score < 7.0:
            analysis['position_issues'] = True
            analysis['position_severity'] = 'high' if pos_score < 3.0 else 'medium' if pos_score < 5.0 else 'low'
        
        # Joint issue analysis
        joint_score = feedback.get('joint_score', 10)
        if joint_score < 7.0:
            analysis['joint_issues'] = True
            analysis['joint_severity'] = 'high' if joint_score < 3.0 else 'medium' if joint_score < 5.0 else 'low'
        
        # Failure reason (if using Joint Critic)
        if 'failure_reason' in feedback and feedback['failure_reason'] != 'success':
            analysis['failure_type'] = feedback['failure_reason']
            
        # Overall rating
        feedback_score = feedback.get('feedback_score', 10)
        analysis['overall_quality'] = 'good' if feedback_score >= 7.0 else 'medium' if feedback_score >= 5.0 else 'poor'
        
        return analysis
    
    def _run_link_with_retry_kwargs(self, prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig, retry_kwargs: Dict):
        """Run Link assembly, pass retry_kwargs to actor"""
        # Modify configuration to include feedback information
        if retry_kwargs and 'previous_feedback' in retry_kwargs:
            # Integrate feedback information into gen_config
            enhanced_gen_config = cfg.gen_config.copy()
            
            # Add feedback-related generation parameters
            if 'improvement_suggestions' in retry_kwargs:
                enhanced_gen_config['feedback_guidance'] = retry_kwargs['improvement_suggestions']
            
            if 'failure_analysis' in retry_kwargs:
                analysis = retry_kwargs['failure_analysis']
                if analysis.get('position_issues'):
                    enhanced_gen_config['focus_positioning'] = True
                if analysis.get('joint_issues'):
                    enhanced_gen_config['focus_joints'] = True
            
            # Create temporary configuration
            temp_cfg = cfg.copy()
            temp_cfg.gen_config = enhanced_gen_config
            
            return articulate_link(prompt, steps, gpu_id, temp_cfg)
        else:
            return articulate_link(prompt, steps, gpu_id, cfg)
    
    def _run_joint_with_retry_kwargs(self, prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig, retry_kwargs: Dict):
        """Run Joint assembly, pass retry_kwargs to actor"""
        # Modify configuration to include feedback information
        if retry_kwargs and 'previous_feedback' in retry_kwargs:
            # Integrate feedback information into gen_config
            enhanced_gen_config = cfg.gen_config.copy()
            
            # Add feedback-related generation parameters
            if 'improvement_suggestions' in retry_kwargs:
                enhanced_gen_config['feedback_guidance'] = retry_kwargs['improvement_suggestions']
            
            if 'failure_analysis' in retry_kwargs:
                analysis = retry_kwargs['failure_analysis']
                # Joint-specific feedback processing
                if 'failure_type' in analysis:
                    enhanced_gen_config['avoid_failure_type'] = analysis['failure_type']
                if analysis.get('joint_issues'):
                    enhanced_gen_config['focus_joint_accuracy'] = True
            
            # Create temporary configuration
            temp_cfg = cfg.copy()
            temp_cfg.gen_config = enhanced_gen_config
            
            return articulate_joint(prompt, steps, gpu_id, temp_cfg)
        else:
            return articulate_joint(prompt, steps, gpu_id, cfg)
    
    def run_basic_pipeline(self) -> Steps:
        """Run basic pipeline (without critic-verifier)"""
        logging.info("Running basic OBJ assembly pipeline...")
        
        # Use default single execution
        result = self.run_single_iteration(0, 0)
        
        if result['success']:
            logging.info("Basic pipeline executed successfully")
        else:
            logging.error(f"Basic pipeline execution failed: {result.get('error', 'Unknown error')}")
        
        return self.steps
    
    def run_critic_verifier_pipeline(self) -> Steps:
        """Run pipeline with Critic-Verifier, implementing true iterative optimization"""
        logging.info("Running Critic-Verifier assembly pipeline...")
        
        try:
            # Initialize critic and verifier
            critic_cfg = self._get_critic_config()
            verifier_cfg = self._get_verifier_config()
            
            critic_type = self.cfg.get('critic_type', 'urdf')
            if critic_type == 'prompt':
                critic = PromptCritic(critic_cfg)
                logging.info("Using PromptCritic for evaluation")
            else:
                critic = URDFCritic(critic_cfg)
                logging.info("Using URDFCritic for evaluation")
            
            verifier = self._create_paper_based_verifier(verifier_cfg)
            logging.info(f"Using paper-method-based Verifier")
            
            # Get configuration parameters
            max_iterations = self.cfg.actor_critic.get('max_iter', 5)
            tau_quality = verifier_cfg.get('quality_threshold', 8.5)
            epsilon_conv = verifier_cfg.get('convergence_threshold', 0.1)
            t_min = verifier_cfg.get('min_iterations', 2)
            
            logging.info(f"Iteration parameters: max_iter={max_iterations}, tau_quality={tau_quality}")
            logging.info("Using fixed seed 0 for deterministic iterative optimization")
            
            # Main iteration loop - using fixed seed
            best_score = -1
            best_iteration = -1
            consecutive_no_improvement = 0
            fixed_seed = 0  # Fixed seed ensures reproducibility
            retry_kwargs = {}  # Store feedback information passed to actor
            
            for iteration in range(max_iterations):
                logging.info(f"Starting iteration {iteration}")
                
                # Prepare feedback (based on historical best results)
                feedback = self._get_feedback_for_iteration(iteration, fixed_seed)
                
                # Actor phase: Generate new URDF, pass retry_kwargs
                actor_result = self.run_single_iteration(iteration, fixed_seed, feedback, retry_kwargs)
                
                if not actor_result.get('success', False):
                    logging.warning(f"Actor failed: iteration {iteration}")
                    consecutive_no_improvement += 1
                    continue
                
                # Find generated URDF file
                urdf_path = self._find_generated_urdf()
                
                if urdf_path and os.path.exists(urdf_path):
                    # Critic phase: Evaluate quality
                    critic_scores = critic.evaluate(urdf_path, iteration, fixed_seed)
                    feedback_score = critic_scores.get('feedback_score', -1)
                    
                    logging.info(f"Critic score: {feedback_score:.3f}")
                    
                    # Save critic results
                    self._save_critic_results(critic_scores, iteration, fixed_seed)
                    
                    # Build structured results
                    result = {
                        'iteration': iteration,
                        'seed': fixed_seed,
                        'feedback_score': feedback_score,
                        'urdf_path': urdf_path,
                        'actor_result': actor_result,
                        **critic_scores
                    }
                    
                    # Update historical records
                    self.history.append(result)
                    self.feedback_history.append(critic_scores)
                    
                    # Update retry_kwargs, use current critic feedback for next iteration
                    retry_kwargs = self._prepare_retry_kwargs_from_critic(critic_scores, iteration)
                    logging.info(f"Preparing retry_kwargs for next iteration: {list(retry_kwargs.keys())}")
                    
                    # Verifier phase: Check termination conditions
                    terminate_1 = feedback_score >= tau_quality  # Quality threshold criterion
                    terminate_2 = False
                    terminate_3 = iteration >= max_iterations - 1  # Maximum iteration count criterion
                    
                    # Convergence criterion
                    if len(self.history) >= 2:
                        prev_score = self.history[-2]['feedback_score']
                        score_diff = abs(feedback_score - prev_score)
                        terminate_2 = (score_diff <= epsilon_conv) and (iteration >= t_min)
                    
                    if terminate_1:
                        logging.info(f"Quality threshold {tau_quality} reached, terminating iteration")
                        return self.steps
                    elif terminate_2:
                        logging.info(f"Score converged (difference {score_diff:.4f} <= {epsilon_conv}), terminating iteration")
                        return self.steps
                    elif terminate_3:
                        logging.info(f"Maximum iteration count {max_iterations} reached, terminating iteration")
                        return self.steps
                    
                    # Check PromptCritic success flag
                    if 'success' in critic_scores and critic_scores['success']:
                        logging.info("PromptCritic evaluation successful, terminating iteration")
                        return self.steps
                    
                    # Record results of this iteration round
                    if feedback_score > best_score:
                        best_score = feedback_score
                        best_iteration = iteration
                        consecutive_no_improvement = 0
                        logging.info(f"Iteration {iteration} achieved best score: {best_score:.3f}")
                    else:
                        consecutive_no_improvement += 1
                        logging.info(f"Iteration {iteration} no improvement, consecutive {consecutive_no_improvement} times")
                        
                else:
                    logging.warning(f"URDF file not found: {urdf_path}")
                    consecutive_no_improvement += 1
                
                # Early stopping mechanism
                if consecutive_no_improvement >= 2 and iteration >= t_min:
                    logging.info(f"No improvement for {consecutive_no_improvement} consecutive times, early stopping")
                    break
            
            logging.info(f"Critic-Verifier pipeline completed, best score: {best_score:.3f} (iteration {best_iteration})")
            return self.steps
            
        except Exception as e:
            logging.error(f"Critic-Verifier pipeline failed: {e}")
            traceback.print_exc()
            logging.warning("Falling back to basic pipeline")
            return self.run_basic_pipeline()
    
    def _get_critic_config(self) -> Dict:
        """Get critic configuration"""
        critic_cfg = self.cfg.get('critic', {})
        critic_type = self.cfg.get('critic_type', 'urdf')
        
        if not critic_cfg:
            logging.warning("Critic configuration not found, using default configuration")
            if critic_type == 'prompt':
                return {
                    'renderer': {
                        'views': ['front', 'side', 'top', 'perspective'],
                        'resolution': [512, 512],
                        'use_sapien': True
                    }
                }
            else:
                return {
                    'renderer': {
                        'views': ['front', 'side', 'top', 'perspective'],
                        'resolution': [512, 512],
                        'use_sapien': True
                    },
                    'scorer': {
                        'position_weight': 0.5,
                        'joint_weight': 0.5,
                        'min_parts_visible': 2,
                        'max_overlap_ratio': 0.3,
                        'min_score_threshold': 5.0,
                        'good_score_threshold': 7.0,
                        'excellent_score_threshold': 8.5
                    }
                }
        return critic_cfg
    
    def _get_verifier_config(self) -> Dict:
        """Get verifier configuration"""
        verifier_cfg = self.cfg.get('verifier', {})
        if not verifier_cfg:
            logging.warning("Verifier configuration not found, using default configuration")
            return {
                'quality_threshold': 8.5,  # tau_quality
                'convergence_threshold': 0.1,  # epsilon_conv
                'min_iterations': 2,  # t_min
                'max_iterations': 5,  # t_max
                'patience': 2
            }
        return verifier_cfg
    
    def _create_paper_based_verifier(self, config: Dict):
        """Create verifier based on paper method"""
        class PaperBasedVerifier:
            def __init__(self, cfg):
                self.config = cfg
                self.tau_quality = cfg.get('quality_threshold', 8.5)
                self.epsilon_conv = cfg.get('convergence_threshold', 0.1)
                self.t_min = cfg.get('min_iterations', 2)
                self.t_max = cfg.get('max_iterations', 5)
        
        return PaperBasedVerifier(config)
    
    def _get_feedback_for_iteration(self, iteration: int, seed: int) -> Dict:
        """Get feedback information for current iteration"""
        if not self.feedback_history:
            return None
        
        # Use latest feedback
        latest_feedback = self.feedback_history[-1].copy()
        
        # Add iteration information
        latest_feedback['iteration'] = iteration
        latest_feedback['seed'] = seed
        
        # Generate detailed feedback text
        pos_score = latest_feedback.get('position_score', 0)
        joint_score = latest_feedback.get('joint_score', 0)
        
        feedback_text = f"Previous iteration scores - Position: {pos_score:.2f}, Joint: {joint_score:.2f}. "
        
        if pos_score < 5.0:
            feedback_text += "Improve component positioning and spatial relationships. "
        if joint_score < 5.0:
            feedback_text += "Improve joint axis alignment and parameter accuracy. "
        
        latest_feedback['detailed_feedback'] = feedback_text
        
        return latest_feedback
    
    def _prepare_retry_kwargs_from_critic(self, critic_scores: Dict, iteration: int) -> Dict:
        """Prepare retry_kwargs for next iteration based on critic scores"""
        retry_kwargs = {}
        
        # Basic feedback information
        retry_kwargs['iteration'] = iteration + 1
        retry_kwargs['previous_scores'] = critic_scores.copy()
        
        # Extract improvement suggestions
        improvement_suggestions = self._extract_improvement_suggestions(critic_scores)
        if improvement_suggestions:
            retry_kwargs['improvement_suggestions'] = improvement_suggestions
        
        # Extract failure analysis
        failure_analysis = self._extract_failure_analysis(critic_scores)
        if failure_analysis:
            retry_kwargs['failure_analysis'] = failure_analysis
        
        # Add specific guidance based on score type
        feedback_score = critic_scores.get('feedback_score', 0)
        pos_score = critic_scores.get('position_score', 0)
        joint_score = critic_scores.get('joint_score', 0)
        
        # Targeted retry guidance
        if pos_score < 5.0:
            retry_kwargs['focus_positioning'] = True
            retry_kwargs['positioning_guidance'] = f"Previous position score: {pos_score:.2f}. Focus on better component placement."
        
        if joint_score < 5.0:
            retry_kwargs['focus_joints'] = True
            retry_kwargs['joint_guidance'] = f"Previous joint score: {joint_score:.2f}. Focus on joint axis and parameter accuracy."
        
        # If specific failure type from Joint Critic
        if 'failure_reason' in critic_scores and critic_scores['failure_reason'] != 'success':
            failure_type = critic_scores['failure_reason']
            retry_kwargs['avoid_failure'] = failure_type
            
            # Give specific guidance based on failure type
            failure_guidance_map = {
                'joint_type': 'Reconsider the joint type (revolute vs prismatic)',
                'joint_axis': 'Check and correct the joint axis direction',
                'joint_origin': 'Adjust the joint pivot point/origin',
                'joint_limit': 'Review and fix joint motion limits and direction'
            }
            
            if failure_type in failure_guidance_map:
                retry_kwargs['failure_specific_guidance'] = failure_guidance_map[failure_type]
        
        # If there are detailed improvement suggestions (from PromptCritic)
        if 'improvement_suggestion' in critic_scores:
            retry_kwargs['detailed_improvement'] = critic_scores['improvement_suggestion']
        
        # Set retry intensity (based on score severity)
        if feedback_score < 3.0:
            retry_kwargs['retry_intensity'] = 'high'
        elif feedback_score < 6.0:
            retry_kwargs['retry_intensity'] = 'medium'
        else:
            retry_kwargs['retry_intensity'] = 'low'
        
        return retry_kwargs
    
    def _find_generated_urdf(self) -> Optional[str]:
        """Find generated URDF file"""
        # Search for URDF files in joint_actor directory
        joint_actor_dir = join_path(self.cfg.out_dir, "joint_actor")
        
        for root, _, files in os.walk(joint_actor_dir):
            for file in files:
                if file.endswith('.urdf'):
                    urdf_path = os.path.join(root, file)
                    logging.info(f"Found URDF file: {urdf_path}")
                    return urdf_path
        
        # Also search in link_placement directory
        link_placement_dir = join_path(self.cfg.out_dir, "link_placement")
        for root, _, files in os.walk(link_placement_dir):
            for file in files:
                if file.endswith('.urdf'):
                    urdf_path = os.path.join(root, file)
                    logging.info(f"Found URDF file: {urdf_path}")
                    return urdf_path
        
        logging.warning("URDF file not found")
        return None
    
    def _save_critic_results(self, critic_scores: Dict, iteration: int, seed: int):
        """Save critic scoring results to critic_renders_iter folder"""
        try:
            # Build critic_renders directory path - directly under joint_actor/iter_0/seed_0
            base_dir = join_path(
                self.cfg.out_dir, 
                "joint_actor", 
                f"iter_{iteration}", 
                f"seed_{seed}"
            )
            
            critic_renders_dir = join_path(base_dir, f"critic_renders_iter{iteration}_seed{seed}")
            
            # Ensure directory exists
            create_dir(critic_renders_dir)
            
            # Save critic results as JSON file
            result_filename = f"prompt_critic_result_iter{iteration}_seed{seed}.json"
            result_path = join_path(critic_renders_dir, result_filename)
            
            # Handle score field mapping - map position_score to pos_score
            processed_scores = dict(critic_scores)
            if 'position_score' in processed_scores and 'pos_score' not in processed_scores:
                processed_scores['pos_score'] = processed_scores['position_score']
                
            # Add additional metadata
            result_data = {
                'iteration': iteration,
                'seed': seed,
                'timestamp': datetime.datetime.now().isoformat(),
                **processed_scores
            }
            
            save_json(result_data, result_path)
            logging.info(f"Critic results saved to: {result_path}")
            
            # Log saved score information
            pos_score = processed_scores.get('pos_score', processed_scores.get('position_score', 0))
            joint_score = processed_scores.get('joint_score', 0)
            feedback_score = processed_scores.get('feedback_score', 0)
            success = processed_scores.get('success', False)
            
            logging.info(f"Saved scores - pos_score: {pos_score}, joint_score: {joint_score}, feedback_score: {feedback_score}, success: {success}")
            
        except Exception as e:
            logging.error(f"Failed to save critic results: {e}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def articulate_obj_assembly(cfg: DictConfig) -> Steps:
    """
    OBJ file-based assembly main function
    
    Usage:
    python articulate_obj_assembly.py prompt="suitcase with a retractable handle" \
        obj_paths.body=/path/to/body.obj \
        obj_paths.handle=/path/to/handle.obj \
        obj_paths.wheel1=/path/to/wheel1.obj
    """
    # Set modality to text (compatible with original system)
    cfg.modality = "text"
    
    # Get obj paths from configuration
    obj_paths = {}
    if hasattr(cfg, 'obj_paths'):
        obj_paths = OmegaConf.to_container(cfg.obj_paths, resolve=True)
    else:
        logging.error("obj_paths configuration not provided")
        raise ValueError("obj_paths configuration must be provided")
    
    if not obj_paths:
        logging.error("obj_paths is empty")
        raise ValueError("obj_paths cannot be empty")
    
    logging.info(f"Prompt: {cfg.prompt}")
    logging.info(f"Provided OBJ files: {list(obj_paths.keys())}")
    logging.info(f"Using Critic-Verifier: {cfg.get('use_critic_verifier', False)}")
    
    # Create and run pipeline
    pipeline = OBJAssemblyPipeline(cfg, obj_paths)
    
    if cfg.get('use_critic_verifier', False):
        return pipeline.run_critic_verifier_pipeline()
    else:
        return pipeline.run_basic_pipeline()


if __name__ == "__main__":
    articulate_obj_assembly()