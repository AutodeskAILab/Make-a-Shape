This is the official codebase for the ICML paper **Make-A-Shape: a Ten-Million-scale 3D Shape Model**
[Project](https://www.research.autodesk.com/publications/generative-ai-make-a-shape/) [Page](https://edward1997104.github.io/make-a-shape/), [ArXiv](https://arxiv.org/abs/2401.11067), [Model](https://github.com/AutodeskAILab/Make-a-Shape), [Demo](https://github.com/AutodeskAILab/Make-a-Shape)

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
