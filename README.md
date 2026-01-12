# DeepTune: Efficient Layer-wise Feature Extraction and Fine-Tuning of Foundation Models

> Collaborative research project exploring optimal strategies for adapting foundation models to downstream vision tasks through efficient layer-wise feature extraction and fine-tuning.

**Authors:** Abdelrahman Werby, Jil Panter, Sejal Jadhav  
**Institution:** University of Freiburg  
**Framework:** PyTorch | AutoML

---

## 🎯 Overview

DeepTune is an AutoML-driven framework for efficiently adapting pre-trained foundation models (DINOv2) to diverse computer vision tasks. The project systematically explores the trade-offs between different fine-tuning strategies, layer selection, and computational efficiency across multiple datasets.


### Computational Resources
- **GPU Hours:** ~1000 hours on RTX 3090
- **Datasets Evaluated:** 4 diverse vision tasks
- **Hyperparameter Configurations:** 10+ strategies tested
- **Fine-tuning Approaches:** Intermediate layers, head-tuning, full fine-tuning

---

## 🏗️ Architecture & Approach

DeepTune leverages **DINOv2** as a foundation model and implements:

1. **Search Space Exploration:**
   - Multiple fine-tuning strategies (LoRA, full fine-tuning, adapter layers)
   - Layer-wise feature extraction optimization
   - Hyperparameter tuning (learning rates, optimizers, schedulers)

2. **Efficient Adaptors:**
   - Selective layer freezing/unfreezing
   - Task-specific head adaptation
   - Computational budget constraints

3. **AutoML Pipeline:**
   - Automated search for optimal layer configurations
   - Performance-efficiency trade-off analysis
   - Time-budget aware model selection

---

## 🚀 Installation

### Conda Environment

To create a conda environment with the necessary dependencies:

```bash
git clone https://github.com/sejal-prog/DeepTune.git
cd DeepTune
conda env create -f env.yaml
conda activate deeptune
```

---

## 📁 Project Structure

```
DeepTune/
├── config/              # Configuration files for experiments
├── deeptune/           # Core implementation
│   ├── models/         # Model architectures
│   ├── search/         # AutoML search logic
│   └── utils/          # Helper functions
├── data/               # Dataset directory (auto-downloaded)
├── env.yaml            # Conda environment
├── tune.py             # Main training script
└── test.py             # Evaluation script
```

---

## 🎯 Usage

### Running Hyperparameter Search

Configure your experiment in `config/deep_tune_config.yaml`, then run:

```bash
python tune.py
```

This will:
- Download datasets automatically (if not present)
- Execute AutoML search across the defined search space
- Save configurations and model checkpoints

### Evaluating Best Configuration

After search completes, update `config/test_config.yaml` with the best configuration number and run:

```bash
python test.py
```

This generates predictions and saves them to `data/exam_dataset/predictions.npy`.

---

## 📊 Datasets

We evaluated DeepTune on 4 diverse datasets:

| Dataset | Classes | Train Samples | Test Samples | Channels | Resolution |
|---------|---------|---------------|--------------|----------|------------|
| Fashion-MNIST | 10 | 60,000 | 10,000 | 1 | 28×28 |
| Flowers | 102 | 5,732 | 2,457 | 3 | 512×512 |
| Emotions | 7 | 28,709 | 7,178 | 1 | 48×48 |
| Skin Cancer | 7 | 7,010 | 3,005 | 3 | 450×450 |

---

## 🔑 Key Findings

1. **Layer-wise feature extraction** from DINOv2 significantly improves efficiency compared to full fine-tuning while maintaining competitive accuracy.

2. **Optimal layer selection** varies by task complexity - shallow tasks benefit from early layers, while complex tasks require deeper features.

3. **Fine-tuning strategy trade-offs:** Full fine-tuning achieves highest accuracy but requires 5-10× more compute than adapter-based approaches.

4. **Search space design matters:** Constraining search to promising regions (based on task characteristics) reduces AutoML time by ~40%.

---

## 🛠️ Technical Stack

- **Framework:** PyTorch
- **Foundation Model:** DINOv2
- **AutoML:** Custom search implementation
- **Optimization:** AdamW, SGD, LoRA, various learning rate schedules
- **Experiment Tracking:** Configuration logging, model checkpointing

---

## 🙏 Acknowledgments

This project was developed as part of the AutoML course at the University of Freiburg.

**Contributors:**
- Abdelrahman Werby
- Jil Panter  
- Sejal Jadhav
rsity of Freiburg}
}
```
