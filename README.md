# T-Cell Receptor-Antigen Interaction Prediction

This project implements a transformer-based model with a novel pretraining strategy for predicting T-cell receptor (TCR) and antigen interactions. The system demonstrates that domain-specific pretraining can improve biological sequence classification performance by **9.96% AUC improvement** over baseline models.

## Project Structure

```
tcr_antigen_prediction/
├── data/                   # Data files (train.csv, test.csv)
├── src/                    # Source code directory
│   ├── data_loader.py     # Data preprocessing and tokenization
│   ├── model.py           # Transformer model implementation
│   ├── pretraining.py     # Novel pretraining strategies
│   ├── training.py        # Model training and fine-tuning
│   └── evaluation.py      # Model evaluation and AUC calculation
├── models/                # Saved model checkpoints
├── results/               # Training results and performance logs
├── main.py               # Complete training pipeline
├── predict.py            # Inference script for trained models
└── requirements.txt      # Python dependencies
```

## Codebase Description

### Core Components

1. **Data Preprocessing** (`src/data_loader.py`):
   - Tokenizes amino acid sequences with special tokens (`<SOS>`, `<SEP>`, `<MASK>`)
   - Creates combined sequences: `<SOS>antigen<SEP>tcr`
   - Handles variable-length sequences with padding/truncation
   - Implements custom collate functions for multi-task pretraining

2. **Transformer Model** (`src/model.py`):
   - 6-layer transformer encoder with 8 attention heads
   - 256-dimensional embeddings with positional encoding
   - Classification head for interaction prediction
   - Multi-task heads for pretraining objectives

3. **Novel Pretraining** (`src/pretraining.py`):
   - **Masked Sequence Modeling**: Predict randomly masked amino acids
   - **Contrastive Learning**: Distinguish related vs unrelated TCR-antigen pairs
   - **Order Prediction**: Learn correct antigen→TCR sequence directionality
   - Multi-task training with task-specific loss functions

4. **Training Pipeline** (`src/training.py`):
   - Baseline model training from scratch
   - Fine-tuning with pretrained weights
   - Early stopping and learning rate scheduling
   - Class-balanced loss functions

5. **Evaluation System** (`src/evaluation.py`):
   - Comprehensive metrics: AUC, accuracy, precision, recall, F1-score
   - Model comparison and improvement analysis
   - Detailed prediction reports and visualizations

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Place your data files in the `data/` directory:
- `train.csv`: Training data with columns (antigen, TCR, interaction)
- `test.csv`: Test data with same format

Example data format:
```csv
antigen,TCR,interaction
GILGFVFTL,CASSSRSSYEQYF,1
NLVPMVATV,CASSPVTGGIYGYTF,1
NLVPMVATV,CASRPDGRETQYF,0
```

## Usage Instructions

### Complete Training Pipeline
Run the full pipeline including pretraining, baseline training, and evaluation:

```bash
python main.py
```

This will:
1. Load and preprocess data (104K+ training, 26K+ test samples)
2. Execute novel pretraining strategy (20 epochs)
3. Train baseline model from scratch (up to 50 epochs)
4. Fine-tune pretrained model (up to 30 epochs)
5. Generate comprehensive performance comparison

### Using Trained Models for Prediction


#### Predict with Baseline Model (No Pretraining)

Model link: 
- option 1: https://drive.google.com/file/d/1QPCOyRHwk9ZMRTHg-7lwsjLg5DBbq1_e/view?usp=sharing
- option 2: https://github.com/SauravUpadhyaya/HW1-Deep-Learning-2025/blob/main/models/baseline_model_best.pt

```bash
python3 predict.py --model baseline --data data/example_new_data.csv --output data/results_unseen_data.csv

```

#### Predict with Pretrained Model

Model link: 
- option 1: https://drive.google.com/file/d/1o91I9zXD-xiA-3LTcfIHf8rQ4dCrYf_A/view?usp=sharing
- option 2: https://github.com/SauravUpadhyaya/HW1-Deep-Learning-2025/blob/main/models/finetuned_model_best.pt


```bash
python3 predict.py --model pretrained --data data/example_new_data.csv --output data/results_unseen_data.csv

```

#### Compare Both Models

```bash
python3 predict.py --model both --data data/example_new_data.csv --output data/results_unseen_data.csv
```

#### Generate ROC Curves
```bash
# ROC curves during prediction
python predict.py --model both --data data/test.csv --evaluate --plot_roc

# Standalone ROC visualization
python visualize_roc.py --baseline_model models/baseline_model_best.pt --pretrained_model models/finetuned_model_best.pt --data_path data/test.csv
```

#### Predict on New Data (No True Labels)
```bash
python predict.py --model pretrained --data new_sequences.csv --output new_predictions.csv
```

## 🚀 **Quick Start: Using Pretrained Model with Your Data**

### Step 1: Prepare Your Input Data

Create a CSV file with your TCR and antigen sequences. The file must have these column names:

```csv
antigen,TCR,interaction
NLVPMVATV,CSALSGNQNYNEQFF,1
TQGYFPDWQNY,CASSYMGQAPYGYTF,0
KLGGALQAK,CASSHQTGDLSYEQYF,0
```

**Important Notes:**
- Use **amino acid single-letter codes** (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- TCR sequences are typically 10-20 amino acids long
- Antigen sequences are typically 8-15 amino acids long
- No spaces or special characters in sequences

### Step 2: Run Prediction

```bash
# Use the best pretrained model for your predictions
python predict.py --model pretrained --data your_data.csv --output results.csv
```

### Step 3: Interpret Results

The output file `results.csv` will contain:

```csv
antigen,TCR,true_interaction,predicted_interaction,no_interaction_prob,interaction_prob,model_used,confidence
AARAVFLAL,CASSYSTGDEQYF,1,0,0.54212874,0.45787126,Pretrained,Medium

**Result Columns:**
- `predicted_interaction`: **1** = Binding predicted, **0** = No binding predicted
- `interaction_prob`: Probability score (0.0 to 1.0, higher = more likely to interact)
- `confidence`: **High** (>=0.7), **Medium** (0.6-0.4), **Low** (<=0.3)

### Example Usage

```bash
# Example 1: Basic prediction
python predict.py --model pretrained --data my_tcr_antigen_pairs.csv --output my_results.csv

# Example 2: If you have true labels and want to evaluate performance
python predict.py --model pretrained --data labeled_data.csv --evaluate --output evaluated_results.csv

# Example 3: Compare baseline vs pretrained models (requires true labels)
python predict.py --model both --data test_data.csv --evaluate --plot_roc
```

### Model Performance

** Recommended: Use Pretrained Model**
- **AUC**: 0.6163 (vs 0.5423 baseline) - **+13.64% improvement**
- **Recall**: 71% improvement in detecting true interactions
- **Best for**: Real-world TCR-antigen interaction screening

### Available Model Files
After training, the following models are available:
- `models/baseline_model_best.pt`: Baseline transformer (no pretraining)
- `models/finetuned_model_best.pt`: **Pretrained + fine-tuned transformer (RECOMMENDED)**
- `models/pretrained_model_final.pt`: Raw pretrained model weights

## Results Summary

### COMPREHENSIVE AUC COMPARISON: BEFORE vs AFTER PRETRAINING

| **Dataset**    | **Before Pretraining** | **After Pretraining** | **Improvement** | **% Change** |
|----------------|------------------------|------------------------|------------------|--------------|
| Training Set   | 0.5427                 | 0.6305                 | +0.0878          | +16.18%      |
| Test Set       | 0.5423                 | 0.6163                 | +0.0740          | +13.64%      |
| **Average**    | –                      | –                      | **+0.0809**      | **+14.91%**  |

---

**Summary**  
- **Training Set Improvement:** +0.0878 (**+16.18%**)  
- **Test Set Improvement:**     +0.0740 (**+13.64%**)  
- **Average Improvement:**      +0.0809 (**+14.91%**)


**Key Achievement**: 9.96% AUC improvement demonstrates successful transfer learning for biological sequence classification.

## Technical Specifications

- **Model Architecture**: 6-layer Transformer encoder
- **Vocabulary**: 25 tokens (20 amino acids + 5 special tokens)
- **Training Device**: CUDA GPU (NVIDIA RTX 4000 Ada recommended)
- **Training Time**: ~3-5 hours for complete pipeline
- **Memory Requirements**: ~8GB GPU memory

## File Descriptions

### Source Code (`src/`)
- `data_loader.py`: Dataset classes and data preprocessing utilities
- `model.py`: Transformer architecture and model definitions  
- `pretraining.py`: Multi-task pretraining implementation
- `training.py`: Training loops for baseline and fine-tuning
- `evaluation.py`: Performance metrics and model comparison tools


## 🛠️ **Troubleshooting & FAQ**

### Common Issues

**Q: "Model file not found" error**
```bash
# Make sure you're in the correct directory
cd tcr_antigen_prediction

# Check if model files exist
ls models/
```

**Q: "Invalid amino acid sequence" error**
- Use only standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- Remove any spaces, numbers, or special characters
- Check for invalid letters like B, J, O, U, X, Z

**Q: "CUDA out of memory" error**
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python predict.py --model pretrained --data your_data.csv --output results.csv
```



### Performance Tips

1. **For best results**: Use sequences within typical length ranges (TCR: 10-20 aa, Antigen: 8-15 aa)
2. **Batch processing**: The model can handle thousands of pairs efficiently
3. **Confidence interpretation**: Focus on High/Medium confidence predictions for important decisions

### System Requirements

- **Python**: 3.8+
- **GPU Memory**: 4GB+ recommended (can run on CPU)
- **RAM**: 8GB+ for large datasets
- **Storage**: 2GB for models and dependencies

This codebase provides a complete framework for TCR-antigen interaction prediction with novel pretraining strategies that significantly improve over baseline approaches.
