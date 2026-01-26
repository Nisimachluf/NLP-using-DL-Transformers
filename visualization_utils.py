import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_utils import clean_text
from utils import load_label_mapping

def plot_class_distribution(dataset_dict, label_mapping_path='classes.json', fname=None):
    """
    Plot class distribution comparison between train, validation, and test splits.
    
    Args:
        dataset_dict: DatasetDict with 'train', 'validation', and 'test' splits
        label_mapping_path: Path to JSON file containing label mappings (default: 'classes.json')
        fname: Optional file path to save the figure (without extension, will be saved as .jpg)
    """
    # Load label mapping
    idx2label, label2idx = load_label_mapping(label_mapping_path)
    
    # Extract labels for all splits
    train_labels = dataset_dict['train']['label']
    val_labels = dataset_dict['validation']['label']
    test_labels = dataset_dict['test']['label']
    
    # Count label occurrences
    train_counts = pd.Series(train_labels).value_counts().sort_index()
    val_counts = pd.Series(val_labels).value_counts().sort_index()
    test_counts = pd.Series(test_labels).value_counts().sort_index()
    
    # Normalize to get proportions
    train_proportions = train_counts / train_counts.sum()
    val_proportions = val_counts / val_counts.sum()
    test_proportions = test_counts / test_counts.sum()
    
    # Prepare data for plotting with actual label names
    label_indices = train_proportions.index.tolist()
    label_names = [idx2label[idx] for idx in label_indices]
    train_values = train_proportions.values
    val_values = val_proportions.values
    test_values = test_proportions.values
    
    # Create bar plot
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width, train_values, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x, val_values, width, label='Validation', alpha=0.8)
    bars3 = ax.bar(x + width, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (show as percentages)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=5.5)
    
    plt.tight_layout()
    
    if fname:
        # Ensure the directory exists
        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{fname}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrices(predictions_cache_path='predictions_cache.json', label_mapping_path='classes.json', 
                           fname=None, selected_model=None, selected_model_fpath=None):
    """
    Plot confusion matrix heatmaps for each model in predictions_cache.json.
    
    Args:
        predictions_cache_path: Path to JSON file containing model predictions (default: 'predictions_cache.json')
        label_mapping_path: Path to JSON file containing label mappings (default: 'classes.json')
        fname: Optional file path to save the entire figure (without extension, will be saved as .jpg)
        selected_model: Optional model name to save separately
        selected_model_fpath: File path to save the selected model's confusion matrix separately (without extension, will be saved as .jpg)
    """
    import json
    from sklearn.metrics import confusion_matrix
    
    # Load predictions cache and label mapping
    with open(predictions_cache_path, 'r') as f:
        predictions_cache = json.load(f)
    
    idx2label, label2idx = load_label_mapping(label_mapping_path)
    label_names = [idx2label[i] for i in sorted(idx2label.keys())]
    
    # Determine number of models
    model_names = list(predictions_cache.keys())
    n_models = len(model_names)
    
    # If selected_model is specified, save it separately first
    if selected_model and selected_model_fpath and selected_model in predictions_cache:
        fig_single, ax_single = plt.subplots(figsize=(4.2, 3.1))
        
        # Get predictions and labels for test set
        test_data = predictions_cache[selected_model]['test']
        y_true = test_data['labels']
        y_pred = test_data['predictions']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by true labels (rows) to get proportions
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names,
                   ax=ax_single, cbar_kws={'label': 'Proportion', 'shrink': 0.85},
                   annot_kws={'fontsize': 7})
        
        ax_single.set_xlabel('Predicted Label', fontsize=9)
        ax_single.set_ylabel('True Label', fontsize=9)
        ax_single.tick_params(axis='both', which='major', labelsize=7.5)
        
        # Adjust colorbar label size
        cbar = ax_single.collections[0].colorbar
        cbar.set_label('Proportion', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.12)
        
        # Save selected model confusion matrix
        directory = os.path.dirname(selected_model_fpath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{selected_model_fpath}.jpg', format='jpg', dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig_single)
    
    # Create subplots for all models - arrange in grid (2 columns)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    # Flatten axes array for easier iteration
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_models > 1 else axes
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        # Get predictions and labels for test set
        test_data = predictions_cache[model_name]['test']
        y_true = test_data['labels']
        y_pred = test_data['predictions']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by true labels (rows) to get proportions
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names,
                   ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if fname:
        # Ensure the directory exists
        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{fname}.jpg', format='jpg', dpi=200, bbox_inches='tight')
    
    plt.show()


def plot_length_distribution(dataset_dict, fname=None):
    """
    Plot text length distribution (in words) for train, validation, and test splits.
    Displays overlapping normalized histograms for comparison.
    
    Args:
        dataset_dict: DatasetDict with 'train', 'validation', and 'test' splits containing 'text' field
        fname: Optional file path to save the figure (without extension, will be saved as .jpg)
    """
    # Calculate word counts for each split
    train_lengths = [len(text.split()) for text in dataset_dict['train']['text']]
    val_lengths = [len(text.split()) for text in dataset_dict['validation']['text']]
    test_lengths = [len(text.split()) for text in dataset_dict['test']['text']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Determine common bins for all plots
    all_lengths = train_lengths + val_lengths + test_lengths
    bins = np.linspace(0, max(all_lengths), 50)
    
    # Plot overlapping histograms with normalization (density=True)
    ax.hist(train_lengths, bins=bins, color='skyblue', edgecolor='black', 
            alpha=0.5, label='Train', density=True)
    ax.hist(val_lengths, bins=bins, color='lightgreen', edgecolor='black', 
            alpha=0.5, label='Validation', density=True)
    ax.hist(test_lengths, bins=bins, color='lightcoral', edgecolor='black', 
            alpha=0.5, label='Test', density=True)
    
    ax.set_xlabel('Text Length (words)', fontsize=12)
    ax.set_ylabel('Normalized Frequency (Density)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics box
    stats_text = f'Train: Mean={np.mean(train_lengths):.1f}, Median={np.median(train_lengths):.1f}\n'
    stats_text += f'Val: Mean={np.mean(val_lengths):.1f}, Median={np.median(val_lengths):.1f}\n'
    stats_text += f'Test: Mean={np.mean(test_lengths):.1f}, Median={np.median(test_lengths):.1f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if fname:
        # Ensure the directory exists
        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{fname}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(trained_weights_dir='trained_weights', fname=None):
    """
    Plot training and validation loss curves for all models in trained_weights directory.
    
    Each subdirectory in trained_weights should contain model folders with training_log.csv files.
    The function plots all models on the same axes with different colors, using solid lines for 
    training loss and dashed lines for validation loss.
    
    Args:
        trained_weights_dir: Path to directory containing trained model weights (default: 'trained_weights')
        fname: Optional file path to save the figure (without extension, will be saved as .jpg)
    """
    # Find all training_log.csv files
    training_logs = []
    for root, dirs, files in os.walk(trained_weights_dir):
        if 'training_log.csv' in files:
            log_path = os.path.join(root, 'training_log.csv')
            # Extract model name from path (e.g., 'google/electra-small-discriminator')
            rel_path = os.path.relpath(root, trained_weights_dir)
            model_name = rel_path.replace(os.sep, '/')
            training_logs.append((model_name, log_path))
    
    if not training_logs:
        print(f"No training_log.csv files found in {trained_weights_dir}")
        return
    
    # Define colors and markers for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_logs)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model
    for idx, (model_name, log_path) in enumerate(training_logs):
        # Read training log
        df = pd.read_csv(log_path)
        
        # Extract epochs and losses
        epochs = df['epoch'].values
        train_loss = df['train_loss'].values
        eval_loss = df['eval_loss'].values
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # Plot training loss (solid line with markers)
        ax.plot(epochs, train_loss, color=color, linestyle='-', marker=marker,
               label=f'{model_name} (train)', linewidth=2, markersize=8, markeredgewidth=1.5,
               markeredgecolor='white')
        
        # Plot validation loss (dashed line with markers)
        ax.plot(epochs, eval_loss, color=color, linestyle='--', marker=marker,
               label=f'{model_name} (val)', linewidth=2, markersize=8, markeredgewidth=1.5,
               markeredgecolor='white')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if fname:
        # Ensure the directory exists
        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{fname}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plot_feature_comparison(pretrained_model_paths, finetuned_model_paths, dataset, 
                            label_mapping_path='classes.json', fname=None, 
                            selected_model=None, selected_model_fpath=None,
                            method='tsne', features_cache_dir='features',
                            metrics_cache_file='feature_separability_metrics.json'):
    """
    Extract features from pretrained and finetuned model pairs, reduce dimensionality to 2D, 
    and create scatter plots colored by label.
    
    Args:
        pretrained_model_paths: List of paths to pretrained models or dict {model_name: model_path}
        finetuned_model_paths: List of paths to finetuned models or dict {model_name: model_path}
        dataset: HuggingFace Dataset with 'text' and 'label' fields
        label_mapping_path: Path to JSON file containing label mappings (default: 'classes.json')
        fname: Optional file path to save the entire figure (without extension, will be saved as .jpg)
        selected_model: Optional model name to save separately (should be in one of the model lists)
        selected_model_fpath: File path to save the selected model's plot separately (without extension)
        method: Dimensionality reduction method - 'tsne', 'pca', or 'umap' (default: 'tsne')
        features_cache_dir: Directory to cache extracted features (default: 'features')
        metrics_cache_file: JSON file to cache all separability metrics (default: 'feature_separability_metrics.json')
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    from sklearn.cluster import KMeans
    from tqdm import tqdm
    
    def extract_model_name(path):
        """Extract model name from path like 'google/electra-small-discriminator' -> 'electra-small'"""
        # Get the last part of the path
        model_part = path.rstrip('/').split('/')[-1]
        # Extract first two parts separated by '-'
        parts = model_part.split('-')
        if len(parts) >= 2:
            return '-'.join(parts[:2])
        return model_part
    
    # Convert to dicts if they're lists
    if isinstance(pretrained_model_paths, list):
        pretrained_dict = {extract_model_name(path): path for path in pretrained_model_paths}
    else:
        pretrained_dict = pretrained_model_paths
    
    if isinstance(finetuned_model_paths, list):
        finetuned_dict = {extract_model_name(path): path for path in finetuned_model_paths}
    else:
        finetuned_dict = finetuned_model_paths
    
    # Ensure same number of models
    if len(pretrained_dict) != len(finetuned_dict):
        raise ValueError("Number of pretrained and finetuned models must match")
    
    # Load label mapping
    idx2label, label2idx = load_label_mapping(label_mapping_path)
    
    # Clean dataset text
    dataset = dataset.map(clean_text)
    
    labels = dataset['label']
    unique_labels = sorted(set(labels))
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metrics cache
    if os.path.exists(metrics_cache_file):
        with open(metrics_cache_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    def get_cache_path(model_name, model_type):
        """Generate cache file path for features"""
        os.makedirs(features_cache_dir, exist_ok=True)
        safe_name = model_name.replace('/', '_')
        return os.path.join(features_cache_dir, f"{safe_name}_{model_type}_{method}.npy")
    
    def save_metrics_to_cache(model_name, model_type, metrics):
        """Save metrics to the global metrics cache file"""
        # Create hierarchy: model-name >> method >> metrics
        model_key = f"{model_name}_{model_type}"
        if model_key not in all_metrics:
            all_metrics[model_key] = {}
        all_metrics[model_key][method] = metrics
        
        # Save to file
        with open(metrics_cache_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def get_metrics_from_cache(model_name, model_type):
        """Get metrics from the global metrics cache file"""
        model_key = f"{model_name}_{model_type}"
        if model_key in all_metrics and method in all_metrics[model_key]:
            return all_metrics[model_key][method]
        return None
    
    def calculate_separability_metrics(features_2d, labels_array):
        """Calculate separability metrics for the reduced features using ground truth labels"""
        # Perform KMeans clustering with the same number of clusters as unique labels
        n_clusters = len(np.unique(labels_array))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_2d)
        
        # Calculate ARI and AMI comparing clusters to ground truth
        metrics = {
            'adjusted_rand_index': float(adjusted_rand_score(labels_array, cluster_labels)),
            'adjusted_mutual_info': float(adjusted_mutual_info_score(labels_array, cluster_labels))
        }
        return metrics
    
    def extract_and_reduce_features(model_path, model_name, model_type="finetuned"):
        """Extract features and apply dimensionality reduction with caching"""
        cache_path = get_cache_path(model_name, model_type)
        
        # Check cache
        if os.path.exists(cache_path):
            print(f"Loading cached {method.upper()} features for {model_name} ({model_type})")
            features_2d = np.load(cache_path)
            
            # Check if metrics are cached
            metrics = get_metrics_from_cache(model_name, model_type)
            if metrics:
                print(f"Loading cached metrics for {model_name} ({model_type})")
            else:
                # Calculate and cache metrics
                print(f"Calculating separability metrics for {model_name} ({model_type})")
                metrics = calculate_separability_metrics(features_2d, np.array(labels))
                save_metrics_to_cache(model_name, model_type, metrics)
            
            return features_2d, metrics
        
        print(f"Extracting features for {model_name} ({model_type})...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to(device)
        
        # Extract features
        features = []
        with torch.no_grad():
            for text in tqdm(dataset['text'], desc=f"Extracting {model_name}"):
                inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                                 max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                features.append(hidden_state.squeeze())
        
        features = np.array(features)
        
        # Apply dimensionality reduction
        print(f"Applying {method.upper()} reduction for {model_name}...")
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = reducer.fit_transform(features)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
        elif method.lower() == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                features_2d = reducer.fit_transform(features)
            except ImportError:
                print("UMAP not installed. Falling back to t-SNE.")
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                features_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'tsne', 'pca', or 'umap'.")
        
        # Cache the results
        np.save(cache_path, features_2d)
        print(f"Cached features to {cache_path}")
        
        # Calculate and cache separability metrics
        print(f"Calculating separability metrics for {model_name} ({model_type})")
        metrics = calculate_separability_metrics(features_2d, np.array(labels))
        save_metrics_to_cache(model_name, model_type, metrics)
        print(f"Cached metrics to {metrics_cache_file}")
        
        return features_2d, metrics
    
    def plot_scatter(ax, features_2d, title, metrics=None):
        """Plot scatter plot on given axes"""
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[color_map[label]], label=idx2label[label], 
                      alpha=0.6, s=25, edgecolors='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=8)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=8)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add metrics text box if provided
        if metrics:
            metrics_text = (
                f"ARI: {metrics['adjusted_rand_index']:.3f}\n"
                f"AMI: {metrics['adjusted_mutual_info']:.3f}"
            )
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Extract features for all model pairs
    model_pairs = []
    selected_pair = None
    
    pretrained_list = list(pretrained_dict.items())
    finetuned_list = list(finetuned_dict.items())
    
    # Find selected model if specified
    selected_model_name = None
    if selected_model:
        selected_model_name = extract_model_name(selected_model)
    
    for (pre_name, pre_path), (fine_name, fine_path) in zip(pretrained_list, finetuned_list):
        # Extract features
        pre_features, pre_metrics = extract_and_reduce_features(pre_path, pre_name, "pretrained")
        fine_features, fine_metrics = extract_and_reduce_features(fine_path, fine_name, "finetuned")
        
        model_pairs.append({
            'pretrained_name': pre_name,
            'finetuned_name': fine_name,
            'pretrained_features': pre_features,
            'finetuned_features': fine_features,
            'pretrained_metrics': pre_metrics,
            'finetuned_metrics': fine_metrics
        })
        
        # Check if this is the selected model pair
        if selected_model_name and (selected_model_name == pre_name or selected_model_name == fine_name):
            selected_pair = model_pairs[-1]
    
    # Print separability metrics summary
    print("\n=== Separability Metrics Summary ===")
    for pair in model_pairs:
        print(f"\nModel: {pair['pretrained_name']}")
        print(f"  Pretrained:")
        print(f"    Adjusted Rand Index: {pair['pretrained_metrics']['adjusted_rand_index']:.4f}")
        print(f"    Adjusted Mutual Info: {pair['pretrained_metrics']['adjusted_mutual_info']:.4f}")
        print(f"  Finetuned:")
        print(f"    Adjusted Rand Index: {pair['finetuned_metrics']['adjusted_rand_index']:.4f}")
        print(f"    Adjusted Mutual Info: {pair['finetuned_metrics']['adjusted_mutual_info']:.4f}")
    
    # If selected_model is specified and found, save it separately
    if selected_model and selected_model_fpath and selected_pair:
        fig_single, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        
        plot_scatter(ax1, selected_pair['pretrained_features'], selected_pair['pretrained_name'], 
                    selected_pair['pretrained_metrics'])
        plot_scatter(ax2, selected_pair['finetuned_features'], selected_pair['finetuned_name'],
                    selected_pair['finetuned_metrics'])
        
        plt.tight_layout()
        
        # Save selected model plot
        directory = os.path.dirname(selected_model_fpath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{selected_model_fpath}.jpg', format='jpg', dpi=200, bbox_inches='tight')
        plt.close(fig_single)
    
    # Create grid plot for all model pairs
    n_pairs = len(model_pairs)
    n_cols = 2
    n_rows = n_pairs
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 4 * n_rows))
    
    # Flatten axes array if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot all pairs
    for idx, pair in enumerate(model_pairs):
        plot_scatter(axes[idx, 0], pair['pretrained_features'], pair['pretrained_name'],
                    pair['pretrained_metrics'])
        plot_scatter(axes[idx, 1], pair['finetuned_features'], pair['finetuned_name'],
                    pair['finetuned_metrics'])
    
    plt.tight_layout()
    
    if fname:
        # Ensure the directory exists
        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{fname}.jpg', format='jpg', dpi=200, bbox_inches='tight')
    
    plt.show()