#!/usr/bin/env python3


import os
import json
import logging
import subprocess
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleSampleURDFGenerator:
    def __init__(self, gpu_id: int = 0):
        """
        Initialize single sample URDF generator
        
        Args:
            gpu_id: GPU ID
        """
        self.gpu_id = gpu_id
    
    def read_semantics_info(self, sample_path: str) -> Tuple[str, Dict[str, Dict]]:
        """
        Read semantics.txt file information
        
        Args:
            sample_path: Sample directory path
            
        Returns:
            Tuple[object_description, link_info_dict]
        """
        sample_dir = Path(sample_path)
        
        # Read semantics file
        semantics_file = sample_dir / "seamantics.txt"
        if not semantics_file.exists():
            semantics_file = sample_dir / "semantics.txt"
            
        if not semantics_file.exists():
            raise FileNotFoundError(f"Cannot find semantics file: {sample_path}")
            
        with open(semantics_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            
        # Parse link information - each line is link info, no object description line
        link_info = {}
        part_types = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse format: link_0 slider drawer
            parts = line.split()
            if len(parts) >= 3:
                link_name = parts[0]  # link_0, link_1, etc.
                joint_type = parts[1]  # slider, free, etc.  
                part_name = ' '.join(parts[2:])  # drawer, furniture_body, etc.
                
                link_info[link_name] = {
                    'joint_type': joint_type,
                    'part_name': part_name
                }
                part_types.add(part_name)
                
        # Generate object description based on part_types
        object_description = f"articulated object with {', '.join(sorted(part_types))}"
                
        return object_description, link_info
    
    def find_obj_files(self, sample_path: str, link_info: Dict) -> Dict[str, str]:
        """
        Find corresponding OBJ files
        
        Args:
            sample_path: Sample path
            link_info: Link information dictionary
            
        Returns:
            OBJ file path dictionary {part_name_with_link_id: obj_path}
        """
        sample_dir = Path(sample_path)
        obj_paths = {}
        
        # Get all OBJ files
        available = {p.name: str(p) for p in sample_dir.glob("*.obj")}
        if not available:
            logger.warning(f"No OBJ files in directory: {sample_dir}")
            return obj_paths

        logger.info(f"Available OBJ files: {sorted(list(available.keys()))}")

        # Process base file (if exists and not empty)
        base_files = ["base_combined_mesh.obj", "base.obj"]
        for base_file in base_files:
            if base_file in available:
                # Check if base file is empty or invalid
                base_path = available[base_file]
                try:
                    with open(base_path, 'r') as f:
                        content = f.read().strip()
                    # Check if there is actual vertex data
                    if content and any(line.startswith('v ') and len(line.split()) >= 4 for line in content.split('\n')):
                        obj_paths["base"] = base_path
                        logger.info(f"Found valid base file: {base_file}")
                        break
                    else:
                        logger.info(f"Skipping empty base file: {base_file}")
                except Exception as e:
                    logger.warning(f"Failed to read base file: {base_file}, error: {e}")

        # Process each link
        for link_id, info in link_info.items():
            idx_match = re.match(r"link_(\d+)$", link_id)
            if not idx_match:
                continue
            i = idx_match.group(1)

            preferred = [
                f"link_{i}_combined_mesh.obj",
                f"link_{i}.obj",
            ]

            chosen_path = None
            # Try preferred names first
            for cand in preferred:
                if cand in available:
                    chosen_path = available[cand]
                    break

            # Match *-link_{i}.obj format
            if chosen_path is None:
                for filename in available.keys():
                    if filename.endswith(f"-link_{i}.obj"):
                        chosen_path = available[filename]
                        logger.info(f"Found matching file: {filename} for {link_id}")
                        break

            # Fallback wildcard link_{i}_*.obj
            if chosen_path is None:
                wildcard = list(sample_dir.glob(f"link_{i}_*.obj"))
                if wildcard:
                    chosen_path = str(wildcard[0])

            if chosen_path is None:
                logger.warning(f"No OBJ file found corresponding to {link_id}")
                continue

            # Use clearer naming: part_name_link_i
            part_name = info['part_name'].replace(' ', '_')
            unique_name = f"{part_name}_{link_id}"
            obj_paths[unique_name] = chosen_path
            logger.info(f"Mapping: {link_id} ({part_name}) -> {unique_name} -> {chosen_path}")
                    
        return obj_paths
    
    def generate_fallback_prompt(self, object_description: str, link_info: Dict) -> str:
        """
        Generate fallback prompt
        
        Args:
            object_description: Object description
            link_info: Link information dictionary
            
        Returns:
            Prompt string
        """
        if link_info:
            # Generate descriptive prompt based on semantic information
            parts_descriptions = []
            for link_id, info in link_info.items():
                joint_type = info['joint_type']
                part_name = info['part_name']
                
                if joint_type == 'slider':
                    parts_descriptions.append(f"sliding {part_name}")
                elif joint_type == 'free':
                    parts_descriptions.append(f"{part_name}")
                else:
                    parts_descriptions.append(f"{joint_type} {part_name}")
                    
            prompt = f"{object_description} with " + " and ".join(parts_descriptions)
        else:
            prompt = object_description
            
        return prompt
    
    def run_urdf_generation(self, prompt: str, obj_paths: Dict[str, str], 
                           sample_name: str, output_dir: str = "single_sample_output") -> Dict:
        """
        Run URDF generation
        
        Args:
            prompt: Input prompt
            obj_paths: OBJ file path dictionary
            sample_name: Sample name
            output_dir: Output directory
            
        Returns:
            Execution result
        """
        # Build output directory
        out_dir = Path(output_dir) / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "articulate_obj_assembly.py",
            f"prompt='{prompt}'",
            f"gpu_id={self.gpu_id}",
            f"out_dir={out_dir}",
            "+use_critic_verifier=true"
        ]
        
        # Add obj path parameters
        for part_name, obj_path in obj_paths.items():
            cmd.append(f"+obj_paths.{part_name}={obj_path}")
            
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run command
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=600,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("URDF generation successful!")
                logger.info(f"Output directory: {out_dir}")
                
                # Find generated URDF files
                urdf_files = list(out_dir.rglob("*.urdf"))
                if urdf_files:
                    logger.info(f"Generated URDF files: {[str(f) for f in urdf_files]}")
                
                return {
                    'success': True,
                    'output_dir': str(out_dir),
                    'urdf_files': [str(f) for f in urdf_files],
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                logger.error(f"URDF generation failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error("URDF generation timeout")
            return {
                'success': False,
                'error': 'Timeout',
            }
        except Exception as e:
            logger.error(f"URDF generation exception: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def process_sample(self, sample_path: str, prompt: Optional[str] = None, 
                      output_dir: str = "single_sample_output") -> Dict:
        """
        Process single sample
        
        Args:
            sample_path: Sample directory path
            prompt: Custom prompt, None means auto-generate
            output_dir: Output directory
            
        Returns:
            Processing result
        """
        sample_dir = Path(sample_path)
        sample_name = sample_dir.name
        
        logger.info(f"Processing sample: {sample_name}")
        logger.info(f"Sample path: {sample_path}")
        
        try:
            # Read semantics information
            object_description, link_info = self.read_semantics_info(sample_path)
            logger.info(f"Object description: {object_description}")
            logger.info(f"Link information: {link_info}")
            
            # Find OBJ files
            obj_paths = self.find_obj_files(sample_path, link_info)
            logger.info(f"Found {len(obj_paths)} OBJ files: {list(obj_paths.keys())}")
            
            if not obj_paths:
                return {
                    'success': False,
                    'error': 'No matching OBJ files found'
                }
            
            # Generate or use provided prompt
            if prompt is None:
                prompt = self.generate_fallback_prompt(object_description, link_info)
                logger.info(f"Auto-generated prompt: {prompt}")
            else:
                logger.info(f"Using custom prompt: {prompt}")
            
            # Run URDF generation
            result = self.run_urdf_generation(prompt, obj_paths, sample_name, output_dir)
            
            # Add additional information to result
            result.update({
                'sample_name': sample_name,
                'sample_path': sample_path,
                'prompt': prompt,
                'obj_paths': obj_paths,
                'object_description': object_description,
                'link_info': link_info
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process sample: {e}")
            return {
                'success': False,
                'error': str(e),
                'sample_name': sample_name,
                'sample_path': sample_path
            }


def main():
    parser = argparse.ArgumentParser(description='Single Sample URDF Generator')
    parser.add_argument('--sample_path', required=True,
                       help='Sample directory path, e.g.: /path/to/47296')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt, auto-generate if not provided')
    parser.add_argument('--auto_prompt', action='store_true',
                       help='Auto-generate prompt (mutually exclusive with --prompt)')
    parser.add_argument('--output_dir', default='single_sample_output',
                       help='Output directory (default: single_sample_output)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (default: 0)')
    
    args = parser.parse_args()
    
    # Check parameters
    if args.prompt and args.auto_prompt:
        parser.error("--prompt and --auto_prompt cannot be used together")
    
    if not os.path.exists(args.sample_path):
        parser.error(f"Sample path does not exist: {args.sample_path}")
    
    # Create generator
    generator = SingleSampleURDFGenerator(gpu_id=args.gpu_id)
    
    # Process sample
    prompt = args.prompt if not args.auto_prompt else None
    result = generator.process_sample(
        sample_path=args.sample_path,
        prompt=prompt,
        output_dir=args.output_dir
    )
    
    # Output results
    print("\n" + "="*50)
    print("Processing Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Sample name: {result['sample_name']}")
        print(f"Used prompt: {result['prompt']}")
        print(f"Output directory: {result['output_dir']}")
        if result.get('urdf_files'):
            print(f"Generated URDF files: {result['urdf_files']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("="*50)


if __name__ == "__main__":
    main()