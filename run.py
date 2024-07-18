
from pathlib import Path
import open3d as o3d

from pytorch_lightning import seed_everything

from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data,
    get_image_transform,
)
from src.model_utils import Model
import argparse


def simplify_mesh(obj_path, target_num_faces=1000):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(target_num_faces)
    o3d.io.write_triangle_mesh(obj_path, simplified_mesh)



checkpoint_path = "checkpoint.ckpt"


def add_args(parser):
    parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="./checkpoint.ckpt",
        # choices=["SV_TO_3D", "MV_TO_3D", "Voxel_TO_3D"], # TODO: Add these models to Hugginface Hub
        help="Model name (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use. If cuda is not available, it will use cpu  (default: %(default)s).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="obj",
        help="Output format (obj, sdf).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Scale of the generated object (default: %(default)s).",
    )
    parser.add_argument(
        "--diffusion_rescale_timestep",
        type=int,
        default=100,
        help="Diffusion rescale timestep (default: %(default)s).",
    )
    parser.add_argument(
        "--target_num_faces",
        type=int,
        default=None,
        help="Target number of faces for mesh simplification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples",
        help="Path to output directory.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    print(f"Loading model")
    model = Model.from_pretrained(pretrained_model_name_or_path=args.model_name)

    if hasattr(model, "image_transform"):
        image_transform = model.image_transform
    else:
        image_transform = None


    data = get_singleview_data(
        image_file=Path(args.image[0]),
        image_transform=image_transform,
        device=model.device,
        image_over_white=False,
    )
    data_idx = 0

    model.set_inference_fusion_params(args.scale, args.diffusion_rescale_timestep)
    output_path = model.test_inference(
        data, data_idx, save_dir=Path(args.output_dir), output_format=args.output_format
    )
