This is the official codebase for **Make-A-Shape**

## Getting Started

### Installation
- Python >= 3.10
- Install CUDA if available
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 
- Install other dependencies by `pip install -r requirements.txt`

For example, on AWS EC2 insatnces with PyTorch Deep learning AMI, you can setup the environment as follows:
```
conda create -n make-a-shape python==3.10
conda activate make-a-shape
pip install -r requirements.txt
```
### Inference
```sh
python run.py examples/jenga.png --model_name SV_TO_3D --output_dir examples --output_format obj 
```
