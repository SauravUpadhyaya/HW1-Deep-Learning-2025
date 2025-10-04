# ROC Curve Visualization Guide

This guide explains how to generate and visualize ROC (Receiver Operating Characteristic) curves for the TCR-antigen interaction prediction models.

## Overview

ROC curves are essential for evaluating binary classification models like our TCR-antigen interaction predictor. They show the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate) at various threshold settings.

## Available ROC Curve Features

### 1. Standalone ROC Visualization Script (`visualize_roc.py`)

This dedicated script generates ROC curves for individual models or comparisons between models.

#### Basic Usage

```bash
# Generate ROC curves for both baseline and pretrained models
python visualize_roc.py --baseline_model models/baseline_model_best.pt --pretrained_model models/finetuned_model_best.pt --data_path data/test.csv

# Generate ROC curve for baseline model only
python visualize_roc.py --baseline_model models/baseline_model_best.pt --data_path data/test.csv

# Generate ROC curve for pretrained model only
python visualize_roc.py --pretrained_model models/finetuned_model_best.pt --data_path data/test.csv

# Custom output directory
python visualize_roc.py --baseline_model models/baseline_model_best.pt --pretrained_model models/finetuned_model_best.pt --data_path data/test.csv --output_dir custom_plots/
```

#### Output Files

The script generates:
- `roc_curve_baseline_model.png` - Individual ROC curve for baseline model
- `roc_curve_pretrained_model.png` - Individual ROC curve for pretrained model  
- `roc_curve_comparison.png` - Side-by-side comparison of both models

### 2. ROC Curves in Main Training Pipeline (`main.py`)

ROC curves are automatically generated during the main training and evaluation pipeline.

After running `python main.py`, check the `results/plots/` directory for:
- `roc_baseline.png`
- `roc_pretrained.png`
- `roc_comparison.png`

### 3. ROC Curves in Prediction Script (`predict.py`)

Generate ROC curves while making predictions using the `--plot_roc` flag.

```bash
# Generate predictions and ROC curves for both models
python predict.py --model both --data data/test.csv --evaluate --plot_roc

# Generate ROC curve for baseline model only
python predict.py --model baseline --data data/test.csv --evaluate --plot_roc

# Generate ROC curve for pretrained model only
python predict.py --model pretrained --data data/test.csv --evaluate --plot_roc
```

## Understanding ROC Curve Components

### Key Elements in Generated ROC Curves

1. **True Positive Rate (TPR) / Sensitivity**: Y-axis
   - Proportion of actual positive cases correctly identified
   - TPR = TP / (TP + FN)

2. **False Positive Rate (FPR)**: X-axis
   - Proportion of actual negative cases incorrectly classified as positive
   - FPR = FP / (FP + TN)

3. **AUC (Area Under Curve)**: Model performance metric
   - AUC = 0.5: Random classifier
   - AUC = 1.0: Perfect classifier
   - Higher AUC = Better model performance

4. **Optimal Point**: Red dot on individual ROC curves
   - Point where TPR - FPR is maximized
   - Represents the best threshold for classification

### ROC Curve Interpretation

- **Curve closer to top-left corner**: Better performance
- **Diagonal line (AUC = 0.5)**: Random classifier baseline
- **Area above diagonal**: Better than random
- **Steeper initial rise**: Better sensitivity at low FPR

## Example Results

Based on our TCR-antigen prediction models:

```
Baseline Model AUC:    0.5423
Pretrained Model AUC:  0.6163
AUC Improvement:       0.0740 (+13.64%)
```

This shows that the pretrained model achieves a **13.64% improvement** in AUC score compared to the baseline, demonstrating the effectiveness of the novel pretraining strategy.

## Technical Implementation Details

### ROC Curve Generation Process

1. **Model Loading**: Load trained model weights
2. **Prediction Generation**: Generate probability scores for test data
3. **ROC Calculation**: Use sklearn.metrics.roc_curve with positive class probabilities
4. **Visualization**: Create matplotlib plots with proper formatting
5. **Comparison**: Overlay multiple models for direct comparison

### Code Integration

ROC curve functionality is integrated into:

- `src/evaluation.py`: Core ROC curve methods
  - `plot_roc_curve()`: Generate individual ROC curves
  - `plot_roc_comparison()`: Compare multiple models
- `visualize_roc.py`: Standalone visualization script
- `predict.py`: ROC curves during prediction
- `main.py`: Automatic ROC generation during training

### Data Requirements

ROC curves require:
- Trained model files (`.pt` format)
- Test data with true labels (`data/test.csv`)
- Proper data format: columns `antigen`, `TCR`, `interaction`

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model paths are correct and models exist
2. **Data Format Issues**: Check CSV columns match expected format
3. **Memory Issues**: Reduce batch size for large datasets
4. **CUDA Issues**: Script automatically handles CPU fallback

### File Locations

- **Models**: `models/baseline_model_best.pt`, `models/finetuned_model_best.pt`
- **Data**: `data/test.csv`
- **Output Plots**: `results/plots/` (default) or custom directory

## Advanced Usage

### Custom Batch Size

```bash
python visualize_roc.py --batch_size 64 --baseline_model models/baseline_model_best.pt --data_path data/test.csv
```

### Specific Device Selection

```bash
python visualize_roc.py --device cuda --baseline_model models/baseline_model_best.pt --data_path data/test.csv
```

### Integration with Custom Analysis

The ROC curve functions return data that can be used for further analysis:

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model, device)
auc_score, fpr, tpr = evaluator.plot_roc_curve(data_loader, save_path='my_roc.png')

# Use fpr, tpr arrays for custom analysis
print(f"AUC Score: {auc_score}")
```

This comprehensive ROC curve functionality provides multiple ways to visualize and analyze the performance of your TCR-antigen interaction prediction models, making it easy to demonstrate the effectiveness of the novel pretraining strategy.
