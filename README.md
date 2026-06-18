<div align="center">

# Physics-Guided Zero-Shot Extended-Depth-of-Field Imaging for Semiconductor Microscopic Inspection


</div>

---

> **Abstract:** >  Microscopic inspection in semiconductor manufacturing is often limited by the shallow depth of field of high-magnification optical systems, which causes dif-ferent regions of the same scene to appear sharp at different focal planes. Multi-focus image fusion can produce all-in-focus images, but many existing learn-ing-based methods depend on supervised training data that are difficult to ob-tain in microscopic industrial settings, while zero-shot approaches may over-smooth fine structures and lose high-frequency defect details. This paper pre-sents a physics-guided zero-shot extended-depth-of-field imaging framework for semiconductor microscopic inspection. The proposed system combines a front-end acquisition and keyframes selection strategy with a back-end unsu-pervised fusion network. During acquisition, a sharpness-driven keyframe se-lection method based on local block evaluation and one-dimensional clustering reduces redundant z-axis frames while retaining representative focus infor-mation. For image fusion, a zero-shot network is designed to exploit the com-plementary sharpness distribution of multi-focus inputs. A Gaussian-guided spatial attention mechanism introduces an explicit defocus prior by modelling the response difference between focused and blurred regions under Gaussian filtering, thereby enhancing edge and texture reconstruction. Experiments on the public MFI-WHU benchmark and a custom semiconductor microscopy da-taset, IC-Focus, suggest that the method improves structural fidelity, reduces artifacts caused by domain shift, and preserves fine microscopic details com-pared with representative traditional, deep learning and deep image prior-based fusion methods. Ablation studies further indicate that the proposed physical prior and gradient-related constraints contribute to reconstruction accuracy and structural consistency. The framework provides a reproducible direction for da-ta-efficient microscopic extended-depth-of-field imaging in industrial visual inspection.



---

## ⚙️ Installation & Requirements
1. **Clone the repository:**

    git clone https://github.com/Teemous/microscopic-fusion.git
    cd microscopic-fusion
   

2. **Install PyTorch:**

    Please install the appropriate PyTorch version matching your CUDA environment:

        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


3. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

##📂    Dataset Preparation
        data_path/
        ├── source_1/ 
        │   ├── 1.jpg
        │   └── 2.jpg
        └── source_2/           
            ├── 1.jpg
            └── 2.jpg
##🚀 Quick Start (Zero-Shot Fusion)
Because our framework utilizes Deep Image Prior, the optimization and fusion happen simultaneously on the fly for each image pair. No pre-training phase is required.

1.**Adjust hyperparameters in ./config/config.yaml.**

2.**Run the fusion script**

    python main.py --config ./config/config.yaml

Output Artifacts:

1.Intermediate focus masks will be saved in mask1/ and mask2/.

2.The final fully focused BGR color images will be saved in the fuse/ directory.

3.The console will output quantitative results including PSNR, SSIM, and No-Reference quality metrics.
