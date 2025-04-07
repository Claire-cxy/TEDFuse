
#  TEDFuse


##  Training

### 1. Virtual Environment

```bash
# Create virtual environment
conda create -n TEDFuse python=3.8.10
conda activate TEDFuse

# Select PyTorch version yourself based on your CUDA version
# Install TEDFuse requirements
pip install -r requirements.txt
```

### 2. Data Preparation

Download the **MSRS dataset**  and place it in the folder:

```bash
./MSRS/
```

### 3. Pre-Processing

Run the following command to generate the training HDF5 file:

```bash
python dataprocessing.py
```

The processed training dataset will be saved at:

```bash
./data/MSRS_train_imgsize_128_stride_200.h5
```

### 4. Training

To train the model, simply run:

```bash
python train.py
```

The trained models will be saved in:

```bash
./models/
```

## 🏄 Testing

### 1. Pretrained Models

We provide pretrained models for both fusion tasks:

- Infrared-Visible Fusion: `./models/fusion.pth`


### 2. Test Datasets

The following test datasets are used in the paper:


- `./test_img/RoadScene`
- `./test_img/TNO`
- `./test_img/M3FD_Fusion`
- `./test_img/droneVehicle`


### 3. Reproducing Results in the Paper


```bash
python test.py
```

