
import base64
import io
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
import numpy as np
import open3d as o3d

from pytorch_lightning import seed_everything
from PIL import Image

from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data,
    get_image_transform
)
from src.model_utils import load_model



def simplify_mesh(obj_path, target_num_faces=1000):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(target_num_faces)
    o3d.io.write_triangle_mesh(obj_path, simplified_mesh)


def _generate_3d_object(
    model_name, model, image_transform, data: Dict[str, Any]
) -> Dict[str, Any]:

    # Get input data
    image_b64 = data.get("image", None)
    images_b64 = data.get("images", None)
    image_views = data.get("image_views", None)
    voxel_b64 = data.get("voxel_npz", None)
    voxel_json = data.get("voxel_json", None)
    output_format = data.get("output_format", "obj")
    text = data.get("text", None)
    diffusion_rescale_timestep = data.get("diffusion_rescale_timestep", 100)
    scale = data.get("scale", 3.0)
    target_num_faces = data.get("target_num_faces", None)
    seed = data.get("seed", 42)

    # Set seed
    seed_everything(seed, workers=True)

    # Run inference
    try:
        if (
            model_name == "Progressive_Diffusion_Generator_SV_TO_3D"
            or model_name == "Progressive_Diffusion_Generator_SV_SKETCH_TO_3D"
        ):
            if not model.args.use_image_conditions:
                return {"error": "Model does not support image conditions"}

            if image_b64:
                image_data = base64.b64decode(image_b64)
                image = io.BytesIO(image_data)
                image_over_white = True if "SKETCH" in model_name else None
                data = get_singleview_data(
                    image, image_transform, model.device, image_over_white
                )
                data_idx = 0
            else:
                return {"error": "Invalid input data"}

        elif model_name == "Progressive_Diffusion_Generator_MV_TO_3D":
            if not model.args.use_image_conditions or not model.args.use_all_views:
                return {"error": "Model does not support multiview conditions"}

            if images_b64 and image_views:
                images = []
                for image_b64 in images_b64:
                    image_data = base64.b64decode(image_b64)
                    image = io.BytesIO(image_data)
                    images.append(image)
                data = get_multiview_data(
                    images, image_views, image_transform, model.device
                )
                data_idx = 0
            else:
                return {"error": "Invalid input data"}

        elif model_name == "Progressive_Diffusion_Generator_VOXEL_TO_3D":
            if not model.args.use_voxel_conditions:
                return {"error": "Model does not support voxel conditions"}

            if voxel_b64:
                voxel_data = base64.b64decode(voxel_b64)
                voxel = io.BytesIO(voxel_data)
                data = get_voxel_data(voxel, model.args.voxel_resolution, model.device)
                data_idx = 0
            elif voxel_json:
                with io.BytesIO() as voxel:
                    np.savez(
                        voxel,
                        occupancy_arr=np.array(voxel_json["occupancy"]),
                        color_numpy=np.array(voxel_json["color"]),
                    )
                    voxel.seek(0)
                    data = get_voxel_data(
                        voxel, model.args.voxel_resolution, model.device
                    )
                data_idx = 0
            else:
                return {"error": "Invalid input data"}

        elif model_name == "MVDREAM_TEXT_TO_MV":
            if text:
                pass
            else:
                return {"error": "Invalid input data"}
        else:
            return {"error": "Invalid model name"}

        if model_name == "MVDREAM_TEXT_TO_MV":
            images_np, image_views = model.inference_step(prompt=text)
            images = [Image.fromarray(image) for image in images_np]
            images_b64 = []
            for image in images:
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    images_b64.append(
                        base64.b64encode(output.getvalue()).decode("utf-8")
                    )

            return {"images": images_b64, "image_views": image_views}
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                save_dir = Path(temp_dir)
                # Set inference fusion parameters
                model.set_inference_fusion_params(scale, diffusion_rescale_timestep)
                # Run inference
                output_path = model.test_inference(
                    data, data_idx, save_dir, output_format=output_format
                )

                if output_format == "sdf":
                    with open(output_path, "rb") as f:
                        sdf_b64 = base64.b64encode(f.read()).decode("utf-8")
                    return {"sdf": sdf_b64, "resolution": model.args.resolution}
                else:
                    if target_num_faces:
                        simplify_mesh(output_path, target_num_faces)

                    with open(output_path, "rb") as f:
                        obj_b64 = base64.b64encode(f.read()).decode("utf-8")

                    return {"obj": obj_b64}

    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        return {"error": str(e)}


model_name = "Progressive_Diffusion_Generator_SV_TO_3D"
checkpoint_path = "checkpoint.ckpt"


device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"CUDA is available: {torch.cuda.is_available()}")

# Load model
print(f"Loading {model_name} model")
model = load_model(
    checkpoint_path,
    compile_model=False,
    device=device,
)
if model.args.use_image_conditions:
    image_transform = get_image_transform(model.args)
else:
    image_transform = None

data = {
    "image": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAACXBIWXMAAAsSAAALEgHS3X78AAAA",
    "output_format": "obj",
    "scale": 3.0,
    "diffusion_rescale_timestep": 100,
    "seed": 42,
}

# data = get_singleview_data(
#     image, image_transform, model.device, image_over_white
# )
# data_idx = 0

# _generate_3d_object(
#     model_name, model, image_transform, data
# )