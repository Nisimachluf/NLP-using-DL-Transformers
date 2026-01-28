# Tweet Emotion Classification with Transformer Models

Project B of the Natural Language Processing Using Deep Learning Techniques course - Bar-Ilan University

## Overview

This project implements emotion classification on English-language tweets using state-of-the-art transformer models. The dataset comprises raw tweets annotated with six emotion classes: **Joy**, **Sadness**, **Fear**, **Anger**, **Love**, and **Surprise**. The project explores multiple transformer architectures and includes model compression techniques to optimize inference speed while maintaining performance.

## Project Summary

This project covers the complete pipeline from data preprocessing to model compression:

#### Data Preprocessing
Raw social media text requires extensive cleaning to remove noise:
- **HTML artifacts removal**: Context-aware sliding window approach to remove HTML tags and metadata
- **Missing apostrophes correction**: Pattern matching to fix corrupted contractions
- **Slang normalization**: Dictionary-based replacement of shorthand text and slang
- **Contraction expansion**: Standardization of contracted forms
- **Character repetition normalization**: Reduction of excessive character repetitions

#### Model Training
Six transformer models of varying sizes were fine-tuned:
- **ELECTRA**: small, base, large (Google)
- **RoBERTa**: base, large (FacebookAI)
- **DeBERTa**: base (Microsoft)

Key training techniques:
- **Hyperparameter tuning** with Optuna (learning rate, epochs, warmup ratio, weight decay)
- **Weighted Cross-Entropy Loss** to handle class imbalance
- **Early stopping** and weight regularization to prevent overfitting
- **Stratified validation split** (20%) by both class and text length

#### Evaluation & Analysis
Models were evaluated using:
- Standard metrics: Accuracy, Precision, Recall, F1-score
- Inference time per sample
- Confusion matrices to identify misclassification patterns
- **Feature separability analysis**: t-SNE visualization + clustering metrics (ARI, AMI)

Key findings:
- ELECTRA-large achieved the best F1 score
- ELECTRA models excel on rare classes despite different pretraining objectives
- RoBERTa-base is ~10% faster than ELECTRA-base (same parameters) due to vocabulary size differences
- DeBERTa is slowest despite fewer parameters (disentangled attention overhead)

#### Model Compression
Compression techniques were explored to reduce model size and improve inference speed:

1. **Single-step approaches**:
   - Pruning: Layer removal with direct training
   - Knowledge Distillation (Student-Teacher): Training smaller model with teacher guidance
   - Hybrid: Combined pruning and distillation

2. **Iterative approach**:
   - Progressive layer pruning (alternating layers: 11, 9, 7, 5, 3, 1)
   - Student becomes teacher for next iteration
   - Evaluated with both CE loss and hybrid loss

Results compare accuracy-speed tradeoffs across all compression strategies.

## Trained Models

All fine-tuned models are available on HuggingFace Hub under `nisimachluf/`:

- `nisimachluf/electra-small-tweet-classification`
- `nisimachluf/electra-base-tweet-classification`
- `nisimachluf/electra-large-tweet-classification`
- `nisimachluf/roberta-base-tweet-classification`
- `nisimachluf/roberta-large-tweet-classification`
- `nisimachluf/deberta-base-tweet-classification`

Compressed models are available under `Yahav/`:
- `Yahav/electra-base-compressed-single-step-hybrid`
- `Yahav/electra-base-compressed-single-step-pruning`
- `Yahav/electra-base-compressed-single-step-ST`
- `Yahav/electra-base-compressed-iterative-approach-pruning`
- `Yahav/electra-base-compressed-iterative-approach-hybrid`

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)

### Setup Virtual Environment

1. **Clone the repository**:
```bash
git clone <repository-url>
cd NLP-using-DL-Transformers
```

2. **Create and activate a virtual environment**:
```bash
python3 -m venv project_b_venv
source project_b_venv/bin/activate  # On Linux/Mac
# or
project_b_venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up HuggingFace authentication** (optional, for model upload):
Create a `.env` file in the project root:
```bash
HF_TOKEN=your_huggingface_token_here
```

## Project Structure

```
.
├── ProjectB.ipynb              # Main notebook: complete pipeline (preprocessing, training, evaluation, compression)
├── data_utils.py              # Data cleaning and preprocessing utilities
├── utils.py                   # Training, evaluation, and helper functions
├── visualization_utils.py      # Plotting and visualization functions
├── hparam_tuning.py           # Hyperparameter optimization with Optuna
├── parse_predictions_to_df.py # Utility to parse results into DataFrame
├── requirements.txt           # Python dependencies
├── classes.json               # Label mapping configuration
├── data/                      # Dataset files
│   ├── train.csv
│   └── validation.csv
├── trained_weights/           # Fine-tuned model checkpoints
├── cached_results/            # Cached predictions and features
├── plots/                     # Generated visualizations
└── hparams/                   # Optimized hyperparameters per model
```

## Usage

Open and run `ProjectB.ipynb` to execute the complete pipeline:
1. Load and clean the tweet dataset
2. Train transformer models with optimized hyperparameters
3. Evaluate models and generate visualizations
4. Compare model performance and feature separability
5. Apply compression techniques (pruning, distillation, hybrid)
6. Analyze accuracy-speed tradeoffs for compressed models
