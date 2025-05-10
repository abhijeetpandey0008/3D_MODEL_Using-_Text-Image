#!/usr/bin/env python3
"""
Combined 3D-Model CLI Tool
This single script integrates functionality from generate.py, preprocess.py,
image_to_model.py, text_to_model.py, utils.py, and visual.py.
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import trimesh
import pyrender
import imageio
from PIL import Image

from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply



# utils.py functionality
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)

# preprocess.py functionality
def load_and_preprocess_image(image_path: str, target_size=(512, 512)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = target_size
    img_resized = cv2.resize(img, (w, h))
    img_tensor = T.ToTensor()(img_resized)
    return img, img_tensor, img_resized.shape[:2]

def remove_background(img_resized):
    mask = np.zeros(img_resized.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (1,1,img_resized.shape[1]-2,img_resized.shape[0]-2)
    cv2.grabCut(img_resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return mask2

def estimate_depth(img_tensor, original_h, original_w, model_type="DPT_Large"):
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform
    input_batch = transform(Image.fromarray((img_tensor.numpy()*255).astype(np.uint8))).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(original_h, original_w), mode="bicubic", align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    return depth

# image_to_model.py functionality
def image_to_mesh(rgb_image, depth_map, mask, scale=1.0):
    h, w = depth_map.shape
    # create point cloud
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    x = (i - w/2) * depth_map / w
    y = -(j - h/2) * depth_map / h
    z = depth_map
    pts = np.stack([x, y, z], axis=-1).reshape(-1,3)
    pts = pts[mask.flatten()==1]
    cloud = trimesh.PointCloud(pts)
    # mesh reconstruction
    try:
        mesh = cloud.convex_hull
    except Exception:
        mesh = trimesh.Trimesh(vertices=pts, process=False)
    mesh.apply_scale(scale)
    return mesh

# text_to_model.py functionality
def prompt_to_mesh(prompt: str ,
                   guidance_scale: float =15.0,
                   num_inference_steps: int =64,
                   frame_size: int = 256,
                   ply_path : str = "temp_mesh.ply"):
     
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     torch_dtype = torch.float16 if (device.type=='cuda') else torch.float32

    # Load the Shap-E pipeline :contentReference[oaicite:7]{index=7}
     pipe = ShapEPipeline.from_pretrained(
        "openai/shap-e",
        torch_dtype=torch_dtype,
        variant="fp16" if device.type=='cuda' else None
    ).to(device)

    # Run inference to get a mesh output :contentReference[oaicite:8]{index=8}
     output = pipe(
        [prompt],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        frame_size=frame_size,
        output_type="mesh"
    )
     
     mesh_raw = output.images[0]  # a trimesh.Trimesh per HF docs :contentReference[oaicite:4]{index=4}
     export_to_ply(mesh_raw, ply_path)                           # :contentReference[oaicite:5]{index=5}
     

     mesh = trimesh.load(ply_path)
     rot = trimesh.transformations.rotation_matrix(
        -np.pi/2, [1, 0, 0]
    )
     mesh.apply_transform(rot)    

    # output.images is a list of trimesh.Trimesh objects :contentReference[oaicite:9]{index=9}
     return mesh
    
    # try:
    #     import clip # type: ignore
    #     _ = clip.load("ViT-B/32", device='cpu')
    # except Exception:
    #     pass
    # # placeholder mesh: unit sphere
    # return trimesh.primitives.Sphere(radius=1.0)

# visual.py functionality
def render_mesh(mesh):
    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    # Position the camera back and above, looking toward the origin
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.707, -0.707, 2.0],
        [0.0,  0.707,  0.707, 2.5],
        [0.0,  0.0,  0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    # Add simple lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)


    viewer = pyrender.Viewer(scene, use_raymond_lighting=False)
    return viewer

def snapshot_mesh(mesh, snapshot_path: str, resolution=(800,800)):
    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    scene.add(camera, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, _ = r.render(scene)
    imageio.imwrite(snapshot_path, color)
    r.delete()
    return snapshot_path

# CLI (generate.py functionality)
def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image or text prompt.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image.")
    group.add_argument("--prompt", type=str, help="Text prompt for mesh generation.")
    parser.add_argument("--snapshot", type=str, help="Path to save snapshot PNG.")
    parser.add_argument("--output", type=str, default="output.obj", help="Path to save OBJ file.")
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.output))

    if args.image:
        img, img_tensor, (h,w) = load_and_preprocess_image(args.image)
        mask = remove_background(img_tensor.numpy().transpose(1,2,0))
        depth = estimate_depth(img_tensor, h, w)
        mesh = image_to_mesh(img, depth, mask)
    else:
        mesh = prompt_to_mesh(args.prompt)


    mesh.export(args.output)
    print(f"Mesh saved to: {args.output}")

    if args.snapshot:
        ensure_dir(os.path.dirname(args.snapshot))
        snapshot_mesh(mesh, args.snapshot)
        print(f"Snapshot saved to: {args.snapshot}")
    else:
        render_mesh(mesh)

if __name__ == "__main__":
    main()
