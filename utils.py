import os
import csv
import json
import torch
import evaluate
import numpy as np 
import os.path as osp
from shutil import copyfile, rmtree
from torch.nn import CrossEntropyLoss
from huggingface_hub import HfApi, errors
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments

from data_utils import preprocess, clean_text

def load_csv_to_dataset(train_path, test_path, label_mapping_path='classes.json'):
    """
    Load CSV files with 'text' and 'label' columns into a HuggingFace Dataset.
    Replaces integer labels with class names from the mapping file.
    
    Args:
        train_path: Path to the training CSV file
        test_path: Path to the test CSV file
        label_mapping_path: Path to JSON file with label mappings (default: 'classes.json')
    
    Returns:
        DatasetDict: HuggingFace DatasetDict with 'train' and 'test' splits
    """
    # Load the dataset
    dataset = load_dataset('csv', data_files={'train': train_path, 'test': test_path})
    
    # Load label mapping
    idx2label, label2idx = load_label_mapping(label_mapping_path)
    
    # Replace integer labels with class names
    def replace_labels(examples):
        examples['label'] = [idx2label[label] for label in examples['label']]
        return examples
    
    dataset = dataset.map(replace_labels, batched=True)
    
    # Cast label column to ClassLabel with the class names
    label_names = [idx2label[i] for i in sorted(idx2label.keys())]
    dataset = dataset.cast_column('label', ClassLabel(names=label_names))
    
    return dataset


def load_label_mapping(json_path='classes.json'):
    """
    Load label mapping from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing label mappings
    
    Returns:
        tuple: (idx2label, label2idx) where:
            - idx2label: Dictionary mapping label indices (int) to label names (str)
            - label2idx: Dictionary mapping label names (str) to label indices (int)
    """
    with open(json_path, 'r') as f:
        idx2label_str = json.load(f)
    
    # Convert string keys to integers for idx2label
    idx2label = {int(k): v for k, v in idx2label_str.items()}
    
    # Create reverse mapping
    label2idx = {v: int(k) for k, v in idx2label_str.items()}
    
    return idx2label, label2idx


def count_parameters(model, verbose=False):
    """
    Count the total number of parameters in a HuggingFace model.
    
    Args:
        model: HuggingFace model
        verbose: If True, prints formatted parameter counts in millions
    
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    if verbose:
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"Non-trainable parameters: {non_trainable_params / 1e6:.2f}M")
    
    return total_params, trainable_params, non_trainable_params


def load_hf_classifier(model_name, n_classes, freeze_backbone=False, training=True):
    """
    Load a HuggingFace model for text classification with a custom classification head.
    
    Args:
        model_name: Name or path of the pretrained model (e.g., 'bert-base-uncased', 'roberta-base')
        n_classes: Number of output classes for classification
        freeze_backbone: If True, only the classifier head is trainable
        training: If False, sets entire model to eval mode with all parameters frozen
    
    Returns:
        model: HuggingFace model with classification head
        tokenizer: Corresponding tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    idx2label, label2idx = load_label_mapping()
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,  # Allows replacing the classification head
        label2id=label2idx,
        id2label=idx2label
    )
    
    if not training:
        # Freeze all parameters and set to eval mode
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    elif freeze_backbone:
        # Freeze all parameters except classifier head
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'score' not in name:
                param.requires_grad = False
    
    return model, tokenizer


def create_model_stats_csv(trained_weights_dir='trained_weights', output_csv='model_stats.csv'):
    """
    Create a CSV file with statistics about models in the trained_weights directory.
    
    Args:
        trained_weights_dir: Path to the trained_weights directory (default: 'trained_weights')
        output_csv: Output CSV file path (default: 'model_stats.csv')
    
    Returns:
        None: Writes the CSV file to disk
    """
    import pandas as pd
    
    stats = []
    
    # Iterate through organization folders
    for org_dir in os.listdir(trained_weights_dir):
        org_path = os.path.join(trained_weights_dir, org_dir)
        if not os.path.isdir(org_path):
            continue
        
        # Iterate through model folders
        for model_dir in os.listdir(org_path):
            model_path = os.path.join(org_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                continue
            
            try:
                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load model to count parameters
                model_name = f"{org_dir}/{model_dir}"
                model, _ = load_hf_classifier(model_path, n_classes=6, training=False)
                total_params, _, _ = count_parameters(model)
                
                # Extract stats from config
                stats.append({
                    'model': model_name,
                    'parameters': total_params,
                    'hidden_size': config.get('hidden_size', None),
                    'num_attention_heads': config.get('num_attention_heads', None),
                    'num_hidden_layers': config.get('num_hidden_layers', None),
                    'vocab_size': config.get('vocab_size', None)
                })
                
            except Exception as e:
                print(f"Error processing {model_path}: {e}")
                continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats)
    df.to_csv(output_csv, index=False)
    print(f"Model statistics saved to {output_csv}")


def predict_on_dataset(model_path, dataset, split='test', batch_size=32, device=None, cache_file='predictions.json'):
    """
    Load a model and make predictions on a dataset split.
    
    Args:
        model_path: Path to the saved model directory
        dataset: HuggingFace Dataset or DatasetDict
        split: Name of the split to predict on (default: 'test')
        batch_size: Batch size for inference (default: 32)
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        cache_file: Path to JSON file for caching results (default: 'predictions.json')
    
    Returns:
        dict: Dictionary containing:
            - 'predictions': list of predicted class indices
            - 'labels': list of true labels (if available)
            - 'metrics': dict with accuracy, recall, precision, f1 scores (if labels available)
            - 'total_time': total inference time in seconds
            - 'time_per_sample_mean': mean time per sample in seconds
            - 'time_per_sample_std': std deviation of time per sample in seconds
    """
    import time
    from tqdm import tqdm
    
    # Extract model name from path for caching
    model_name = "-".join(model_path.split('/')[-1].split("-")[:2])
    cache_key = f"{model_name}"
    
    # Check if results are already cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache_key in cache and split in cache[cache_key]:
            print(f"Loading cached results for {cache_key} on {split} split")
            return cache[cache_key][split]
    else:
        cache = {}
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    print(f"Loading model: {cache_key}")
    model, tokenizer = load_hf_classifier(model_path, n_classes=6, training=False)
    model.to(device)
    
    # Get the dataset split
    if isinstance(dataset, DatasetDict):
        data = dataset[split]
    else:
        data = dataset
    data = data.map(clean_text)
    
    # Prepare for inference
    all_predictions = []
    all_probabilities = []
    all_labels = []
    sample_times = []
    
    # Track inference time
    start_time = time.time()
    
    # Process in batches with progress bar
    num_batches = (len(data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Inference"):
        batch_end = min(i + batch_size, len(data))
        batch_texts = data['text'][i:batch_end]
        batch_size_actual = batch_end - i
        
        # Tokenize batch with padding to longest in batch
        tokenized = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in tokenized.items()}
        
        # Get predictions - time only the model inference
        with torch.no_grad():
            batch_start_time = time.time()
            outputs = model(**inputs)
            batch_time = time.time() - batch_start_time
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(preds.cpu().numpy().tolist())
        all_probabilities.extend(probs.cpu().numpy())
        
        # Collect labels if available
        if 'label' in data.features:
            all_labels.extend(data['label'][i:batch_end])
        
        # Track time per sample in this batch
        sample_times.extend([batch_time / batch_size_actual] * batch_size_actual)
    
    # Calculate total time and time per sample statistics
    total_time = time.time() - start_time
    time_per_sample_mean = np.mean(sample_times)
    time_per_sample_std = np.std(sample_times)
    
    result = {
        'predictions': all_predictions,
        'total_time': total_time,
        'time_per_sample_mean': time_per_sample_mean,
        'time_per_sample_std': time_per_sample_std
    }
    
    if all_labels:
        result['labels'] = all_labels
        
        # Calculate metrics if labels are available
        compute_metrics = get_metrics_fn()
        metrics = compute_metrics((np.array(all_probabilities), np.array(all_labels)))
        result['metrics'] = metrics
    
    # Save to cache with compact format for lists
    print(f"Saving results to {cache_file}")
    if cache_key not in cache:
        cache[cache_key] = {}
    cache[cache_key][split] = result
    cache_dir = osp.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2, separators=(',', ': '))
    
    # Rewrite with compact lists
    with open(cache_file, 'r') as f:
        content = f.read()
    
    # Replace multiline lists with single-line compact format
    import re
    content = re.sub(
        r'"(predictions|labels)":\s*\[\s*((?:\d+,?\s*)+)\]',
        lambda m: f'"{m.group(1)}": [{", ".join(m.group(2).replace(",", "").split())}]',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    with open(cache_file, 'w') as f:
        f.write(content)
    
    return result

class CSVLoggingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_log.csv")
        print("Logging training results to:", self.filepath)
        self.initialized = False
        self.fieldnames = ["epoch", "step", "loss", "eval_loss", "eval_accuracy", "eval_recall", "eval_precision", "eval_f1"]
        self.buffer = {}
    
    def is_full(self):
        return all([key in self.buffer for key in self.fieldnames])
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # Add step info
        logs = dict(logs)
        logs["step"] = state.global_step
        self.buffer.update(logs)
        print(self.buffer)
        if self.is_full():
            write_header = not os.path.exists(self.filepath)
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if write_header:
                    writer.writeheader()
                cleaned_buffer = {k: self.buffer[k] for k in self.fieldnames}
                print("writing row:", cleaned_buffer)
                writer.writerow(cleaned_buffer)
            self.buffer.clear()
            
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.class_weights = torch.tensor(class_weights)
        self.loss_fun = None
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.loss_fun is None:
            self.loss_fun = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = self.loss_fun(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
def calc_class_weights(dataset):
    counts = np.bincount(dataset["label"])
    weights = (counts.max() / counts).astype(np.float32)
    return weights

def get_metrics_fn():
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1_score = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {}
        metrics.update(accuracy.compute(predictions=predictions, references=labels))
        metrics.update(recall.compute(predictions=predictions, references=labels, average="macro") )
        metrics.update(precision.compute(predictions=predictions, references=labels, average="macro"))
        metrics.update(f1_score.compute(predictions=predictions, references=labels, average="macro"))
        return metrics

    return compute_metrics


def train(model_name, dataset, output_dir, learning_rate, batch_size, num_train_epochs, log_results, early_stopping_patience=3, lr_scheduler_type="linear", warmup_ratio=0.05, weight_decay=0.01, use_weighted_loss=True, save_trained_model=True, trained_weights_dir="trained_weights"):
    
    output_dir = osp.join(output_dir, model_name)
    if osp.isdir(output_dir):
        rmtree(output_dir)
    
    model, tokenizer = load_hf_classifier(model_name, 6, freeze_backbone=False)

    processed_dataset = preprocess(dataset, tokenizer)
    
    
    compute_metrics = get_metrics_fn()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,        
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        push_to_hub=False,

    )
    
    callbacks = []
    if log_results:
        callbacks.append(CSVLoggingCallback(output_dir))
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        
        
    trainer_args = {"model": model,
                    "args": training_args,
                    "train_dataset": processed_dataset["train"],
                    "eval_dataset": processed_dataset["validation"],
                    "processing_class": tokenizer,
                    "data_collator": data_collator,
                    "compute_metrics": compute_metrics,
                    "callbacks": callbacks}
    
    trainer_cls = Trainer
    if use_weighted_loss:
        trainer_args["class_weights"] = calc_class_weights(processed_dataset["train"])
        trainer_cls = WeightedTrainer
        
    trainer = trainer_cls(**trainer_args)
    
    trainer.train()
    
    if save_trained_model:
        trained_model_path = osp.join(trained_weights_dir, model_name)
        os.makedirs(trained_weights_dir, exist_ok=True)
        trainer.save_model(trained_model_path)
        if log_results:
            log_fname = "training_log.csv"
            copyfile(osp.join(output_dir, log_fname), osp.join(trained_model_path, log_fname))
                    
    return trainer

def model_exists_on_hub(model_id):
    """
    Checks if a model with the given model_id exists on the Hugging Face Hub.
    """
    api = HfApi()
    try:
        api.model_info(model_id)
        return True
    except errors.RepositoryNotFoundError:
        return False
    except Exception as e:
        # Handle other potential errors (e.g., network issues, authentication errors)
        print(type(e))
        print(f"An error occurred: {e}")
        return False
