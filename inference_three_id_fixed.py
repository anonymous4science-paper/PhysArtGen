#!/usr/bin/env python3

import argparse
import os
import torch
import numpy as np
import trimesh
import mcubes

# Import fixed models
import models_three_id_fixed
import models_ae

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--model', default='three_id_kl_d512_m512_l8_edm', help='Model name')
    parser.add_argument('--model_pth', required=True, help='Model weights path')
    parser.add_argument('--ae', default='kl_d512_m512_l8', help='AutoEncoder model')
    parser.add_argument('--ae_pth', required=True, help='AutoEncoder weights path')
    
    # Condition parameters
    parser.add_argument('--major_category', default=0, type=int, help='Major category ID (0=drawer)')
    parser.add_argument('--part_category', default=0, type=int, help='Part category ID (0=body, 1=door, 2=slider-drawer)')
    parser.add_argument('--identifier', required=True, type=str, help='Identifier string')
    
    # Generation parameters
    parser.add_argument('--num_samples', default=1, type=int, help='Number of samples to generate')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--density', default=64, type=int, help='Voxel density')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./inference_output_fixed', help='Output directory')
    parser.add_argument('--output_format', default='obj', choices=['obj', 'ply'], help='Output format')
    
    # Other
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--identifier_mapping_file', type=str,
                       default='/home/zhaochaoyang/yuantingyu/3DShape2vecset/data_drawer/identifier_mapping.json',
                       help='Identifier mapping file')
    
    return parser.parse_args()

def create_volume_grid(density=64, device='cuda'):
    """Create 3D voxel grid, following original code format"""
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)
    # grid shape: [1, (density+1)^3, 3]
    return grid

def decode_latent_to_volume(ae, latent_codes, density=64, device='cuda'):
    """Decode latent codes to 3D voxels"""
    # Create query grid
    grid = create_volume_grid(density, device)  # [1, (density+1)^3, 3]
    
    num_latents = latent_codes.shape[0]
    volumes = []
    
    for i in range(num_latents):
        latent = latent_codes[i:i+1]  # [1, N, C]
        
        with torch.no_grad():
            # Decode - following original code format
            logits = ae.decode(latent.float(), grid)
            logits = logits.detach()
            
            # Reshape to 3D volume, following original code format
            volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
            volumes.append(volume)
    
    return volumes  # List of numpy arrays

def extract_mesh_marching_cubes(volume, density, threshold=0.0):
    """Extract mesh using marching cubes"""
    try:
        # Extract surface using marching cubes
        vertices, faces = mcubes.marching_cubes(volume, threshold)
        
        # Scaling according to original code
        gap = 2. / density
        vertices *= gap
        vertices -= 1
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None

def main():
    args = get_args()
    
    print(f"🎯 Fixed Three-ID Conditional 3D Generation Inference")
    print(f"📦 Model: {args.model}")
    print(f"💾 Weights: {args.model_pth}")
    print(f"🎨 Conditions: major_category={args.major_category}, part={args.part_category}, identifier='{args.identifier}'")
    print(f"🔢 Generation count: {args.num_samples}")
    
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load AutoEncoder
    print("Loading AutoEncoder...")
    if args.ae == 'kl_d512_m512_l8':
        ae = models_ae.KLAutoEncoder(
            dim=512, 
            num_latents=512, 
            latent_dim=8, 
            num_inputs=2048
        )
    else:
        raise ValueError(f"Unknown AE model: {args.ae}")
    
    ae_checkpoint = torch.load(args.ae_pth, map_location='cpu')
    ae.load_state_dict(ae_checkpoint['model'])
    ae.to(device)
    ae.eval()
    
    # Load generation model
    print("Loading generation model...")
    if args.model == 'three_id_kl_d512_m512_l8_edm':
        model = models_three_id_fixed.three_id_kl_d512_m512_l8_edm(
            identifier_mapping_file=args.identifier_mapping_file
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    checkpoint = torch.load(args.model_pth, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loading completed")
    
    # Prepare conditions
    major_category_ids = torch.tensor([args.major_category] * args.num_samples).long().to(device)
    part_category_ids = torch.tensor([args.part_category] * args.num_samples).long().to(device)
    identifiers = [args.identifier] * args.num_samples
    batch_seeds = torch.arange(args.seed, args.seed + args.num_samples).to(device)
    
    print(f"Generating samples...")
    
    # Generate latent codes
    with torch.no_grad():
        latent_codes = model.sample(
            major_category_ids=major_category_ids,
            part_category_ids=part_category_ids,
            identifiers=identifiers,
            batch_seeds=batch_seeds
        )
    
    print(f"✅ Generated latent codes: {latent_codes.shape}")
    
    # Decode to 3D voxels
    print(f"Decoding to 3D voxels...")
    volumes = decode_latent_to_volume(ae, latent_codes, args.density, device)
    
    print(f"✅ Decoding completed: {len(volumes)} volumes, each shape {volumes[0].shape}")
    
    # Extract meshes and save
    print(f"Extracting meshes...")
    for i in range(args.num_samples):
        volume = volumes[i]  # numpy array
        
        # Extract mesh using marching cubes
        mesh = extract_mesh_marching_cubes(volume, args.density)
        
        if mesh is not None and len(mesh.vertices) > 0:
            # Construct filename
            filename = f"generated_major{args.major_category}_part{args.part_category}_id{args.identifier}_sample{i}.{args.output_format}"
            output_path = os.path.join(args.output_dir, filename)
            
            # Save mesh
            mesh.export(output_path)
            print(f"✅ Saved: {output_path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
        else:
            print(f"❌ Sample {i}: mesh extraction failed")
    
    print(f"🎉 Inference completed! Results saved in {args.output_dir}")

if __name__ == '__main__':
    main() 