# PhysArtGen

PhysArtGen is a codebase for articulated 3D object understanding and generation with physics-aware evaluation. It combines:

- Three-ID conditional latent diffusion for 3D representations (training in `main_three_id_fixed.py`, model in `models_three_id_fixed.py`).
- AutoEncoder for point-cloud latent encoding (`models_ae.py`, training utils in `engine_ae.py`).
- Multimodal agent framework for mesh retrieval, link placement, joint prediction, and verification (`articulated_utils/agent/*`).
- Physics simulation and rendering utilities (PyBullet/SAPIEN) in `articulated_utils/physics/*` with URDF tools in `articulated_utils/api/*`.

---

## Repository Structure

- `main_three_id_fixed.py`: Training entry for the Three-ID conditional EDM model.
- `engine_three_id.py`: Training/eval loops for the Three-ID model.
- `models_three_id_fixed.py`: Model definitions (EDM preconditioning, Transformer, Three-ID embedding).
- `models_ae.py` and `engine_ae.py`: AutoEncoder and its training/eval utilities.
- `articulated_utils/agent/`: Agent framework with actor/critic/verifier modules.
  - `actor/mesh_retrieval`, `actor/link_placement`, `actor/joint_prediction`
  - `critic/` for joint/link/URDF critics; `verifier/` for stopping logic
- `articulated_utils/api/`: URDF creation utilities.
- `articulated_utils/physics/`: PyBullet/SAPIEN render/sim helpers.
- `articulated_utils/utils/`: CLIP utilities, meshing, metrics, prompts, visualization, etc.
- `preprocess/`: Scripts to prepare datasets and embeddings (e.g., PartNet annotations/embeddings).
- `baselines/real2code/`: Baseline dataset and training/eval utilities.
- `articulated-class/`: Example meshes and metadata organized by category.
- `inference_three_id_fixed.py`, `single_sample_urdf_generator.py`, `urdf_video_simulator.py`: Inference and visualization helpers.

---

## Requirements

- Python 3.9+
- PyTorch (+ CUDA if using GPU)
- timm
- numpy
- omegaconf
- PyBullet and/or SAPIEN (for physics/rendering)
- tensorboard (optional)

Install essentials (example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm omegaconf numpy tensorboard pybullet sapien
```

Note: adjust CUDA/PyTorch wheels per your environment.

---

## Data

Training expects a dataset of articulated objects with point clouds and identifiers (see `util/datasets.py` and `util/articulated_dataset.py`). Example assets are under `articulated-class/`. Preprocessing utilities are in `preprocess/` (e.g., `preprocess_partnet.py`, `create_partnet_embeddings.py`).

You will also need an identifier mapping JSON for Three-ID (see `models_three_id_fixed.py`, argument `--identifier_mapping_file`).

AutoEncoder weights are required by the Three-ID training (`--ae_pth`). Train your AE first or use your existing checkpoint.

---

## Train the Three-ID Conditional Model

Example command (single GPU):

```bash
python main_three_id_fixed.py \
  --data_path /path/to/dataset \
  --ae_pth /path/to/ae_checkpoint.pth \
  --identifier_mapping_file /path/to/identifier_mapping.json \
  --batch_size 128 --epochs 200 --output_dir ./output_three_id_fixed
```

Notes:
- Distributed training is auto-detected if `LOCAL_RANK` is set (e.g., using `torchrun`).
- Logs and checkpoints are written to `--output_dir`.

Distributed (example):

```bash
torchrun --nproc_per_node=4 main_three_id_fixed.py \
  --data_path /path/to/dataset \
  --ae_pth /path/to/ae_checkpoint.pth \
  --identifier_mapping_file /path/to/identifier_mapping.json
```

---

## AutoEncoder

Train/evaluate the AutoEncoder using `engine_ae.py` utilities (see how losses and IoU are computed). Integrate its checkpoint via `--ae_pth` for Three-ID training. The AE encodes surface point clouds into latent arrays consumed by the diffusion model.

---

## Inference and Visualization

- `inference_three_id_fixed.py`: Sampling utilities for the Three-ID model (provide IDs and seeds).
- `single_sample_urdf_generator.py`: Build URDFs from predicted parts.
- `urdf_video_simulator.py`: Simulate and render articulated motion videos.
- `articulated_utils/api/odio_urdf.py`: Programmatic URDF construction helpers.

Physics/render backends:
- PyBullet helpers in `articulated_utils/physics/pybullet_utils.py`.
- SAPIEN helpers in `articulated_utils/physics/sapien_render.py` and `sapien_simulate.py`.

---

## Multimodal Agent (Optional Workflow)

Under `articulated_utils/agent/`, the agent system orchestrates:
- Mesh retrieval from a library (`actor/mesh_retrieval/`),
- Link placement (`actor/link_placement/`),
- Joint prediction (`actor/joint_prediction/`),
- Critique and verification (`critic/*`, `verifier/*`).

Prompts are documented in `agent_prompts.md`.

---

## License

This repository is for research purposes. See the paper for any additional terms or dataset licenses.
