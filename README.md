# DD2412_DLA_FinalProject

**Project Conducted by:** Jacob Lundgren (*jaclundg@kth.se*), Gustav Kopp Sundin (*gwks@kth.se*), and Ludvig Karlsson (*ludkar@kth.se*)

## Project Overview

This project is a part of the course DD2412 Deep Learning, Advanced Course at KTH Royal Institute of Technology. The project reproduces and extends the findings from the paper **"Class-Conditional Conformal Prediction with Many Classes"** by Tiffany Ding et al., which focuses on **uncertainty estimation in deep networks** using conformal prediction techniques. Specifically, the paper introduces **Clustered Conformal Prediction (CCP)**, a novel approach to achieving class-conditional coverage in multi-class classification problems with limited data.

### Objectives
1. **Reproduce the results** of the original paper, particularly focusing on CCP and its comparison to standard conformal prediction (CP) and classwise conformal prediction.
2. **Extend the original methodology** by exploring the impact of alternative clustering algorithms (Gaussian Mixture Models and Hierarchical Clustering) on CCP performance.

## Methodology

### Dataset
- **CIFAR-100**: The experiments used the CIFAR-100 dataset, consisting of 60,000 images across 100 classes. The dataset was re-partitioned into fine-tuning and calibration subsets.

### Core Implementation Steps
1. **Fine-Tuning**: A ResNet-50 model, pretrained with IMAGENET1K_V2 weights, was fine-tuned on a subset of CIFAR-100 for 30 epochs, achieving ~59% accuracy.
2. **Embedding Generation**: Quantile-based embeddings were created using the methodology described in the original paper, leveraging its code for consistency.
3. **Clustering**: Class embeddings were grouped using three clustering methods:
   - **Weighted K-Means** (baseline)
   - **Gaussian Mixture Models (GMM)**
   - **Agglomerative Clustering (AC)**
4. **Evaluation**: Performance was measured using two metrics:
   - **CovGap**: Deviation of coverage from the target level.
   - **AvgSize**: Average size of prediction sets.
  
## Results and Findings

1. **Reproduction Results**:
   - Successfully replicated the results for standard and classwise CP.
   - Small deviations were observed in CCP results.
   - Performance consistency across data regimes differed, with clustering methods impacting results only in high-data regimes.

2. **Extended Findings**:
   - GMM and AC clustering methods were explored as alternatives to K-Means.
   - No significant impact was observed in low-data regimes, but clear differences emerged with increasing calibration samples.

## How to Run
### 1. Clone the Repository
```bash
git clone <https://github.com/Jaken1735/DD2412_DLA_FinalProject>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset

- The project uses the CIFAR-100 dataset, which will be automatically downloaded via PyTorch when running the code.
- Alternatively, download CIFAR-100 manually and place it in the appropriate directory. Update the dataset path in the configuration file if necessary.

### 4. Run the Code
**Fine-Tune Classifier (Optional):**
```bash
python training.py
```

**Conformal Prediction:**
```bash
sh main.sh
```
