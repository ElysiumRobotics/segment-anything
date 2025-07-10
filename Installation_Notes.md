# Segment Anything Installation at Elysium 

Follow: [Elysium's Segment Anything Fork](https://github.com/ElysiumRobotics/segment-anything)
Original: [Meta's Segment Anything](https://github.com/facebookresearch/segment-anything)

## Install core dependencies
In lightning
'
sudo apt install libopencv-dev
sudo apt install libxcb-cursor0
'

## Installed CUDA

### Check for NVIDA GPU
'lspci | grep -i nvidia'

### Install NVIDIA's CUDA Toolkit and Drivers
[NVIDIA's Instructions](
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

```
mkdir -p ~/Documents/temp_cuda; cd ~/Documents/temp_cuda/
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

sudo apt-get install -y cuda-drivers
```

Following nvidia instructions I installed and verified cuda-12.9 (previously 12.1)

Test it: 
```
nvidia-smi
```
I See:
```
NVIDIA-SMI 575.57.08        
Driver Version: 575.57.08
CUDA Version: 12.9
```

## Setup Environment and Install PyTorch
```
mkdir ~/Documents/Projects/microFibers/segment_anything
conda create -n segment_anything python=3.10
conda activate segment_anything
```

Use [PyTorch Configurator](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # This worked!!!
```

Test PyTorch:
```
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
I see:
```
CUDA: 12.8, Available: True, GPU: Quadro RTX 4000
```

## Back to Segment Anything
Test it
```
pip install -e . # Add segment anything to path

python scripts/amg.py --checkpoint model/sam_vit_h_4b8939.pth --model-type vit_h --input /home/Projects/uFibers/Elysium_Micrographs/2025/runs/run_202505xx/spool_300/20230901/cross-section/Fiber_1_I.jpg  --output output/
```
I see:
```
Done
```
