## Model Architecture

The HG-LGBM model consists of several key components:

- **Heterogeneous Graph Transformer (HGT)**: Leverages multi-head attention mechanisms to process different types of biological entities and their relationships
- **Feature Extraction**: Extracts meaningful representations for microbes and diseases
- **Association Prediction**: Calculates importance scores between microbes and diseases using feature similarity
- **Ensemble Learning**: Integrates predictions from multiple models to improve accuracy

## Key Features

- Integration of diverse biological data sources
- Heterogeneous graph-based representation learning
- Cross-validation for robust performance evaluation
- Importance score calculation for microbe-disease associations
- High performance metrics (AUC, AUPR, F1-score, etc.)

## Usage

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- scikit-learn
- pandas
- numpy
- joblib

### Running the Model

```bash
# Train the model with default parameters
python main.py

# Train with custom parameters
python main.py --dataset HMDAD --hidden_channels 512 --num_heads 8 --num_layers 6 --epochs 200

# Calculate importance scores between microbes and diseases
python calculate_importance.py --dataset HMDAD
```

### Analyzing Microbe-Disease Importance

To analyze the importance scores between microbes and diseases:

```bash
python calculate_importance.py --dataset HMDAD
```

This will generate multiple CSV files in the `[dataset]/importance_results/` directory:
- `all_md_importance_scores.csv`: All microbe-disease association scores
- `top_microbe_disease_associations.csv`: Top-ranked associations
- `top_microbes_per_disease.csv`: Most relevant microbes for each disease
- `top_diseases_per_microbe.csv`: Most relevant diseases for each microbe

## Datasets

The model supports multiple datasets:
- HMDAD (Human Microbe-Disease Association Database)
- Disbiome

Each dataset includes:
- Microbe-disease known associations
- Microbe features
- Disease features
