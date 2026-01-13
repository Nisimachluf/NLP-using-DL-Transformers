import os
import csv
import json
import torch
import evaluate
import numpy as np 
import os.path as osp
from shutil import copyfile, rmtree
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments

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


def preprocess(dataset, tokenizer, validation_size=0.2, random_state=42):
    """
    Preprocess dataset by adding token length column and creating stratified train/validation splits.
    
    Args:
        dataset: DatasetDict from load_csv_to_dataset
        tokenizer: HuggingFace tokenizer
        validation_size: Proportion of training data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        DatasetDict: Processed dataset with 'train', 'validation', and 'test' splits
    """
    
    # Add length column to all splits
    def add_length(examples):
        examples['length'] = [len(tokenizer.encode(text)) for text in examples['text']]
        return examples
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    dataset = dataset.map(add_length, batched=True)
    
    # Create stratified train/validation split based on label and length bins
    train_dataset = dataset['train']
    
    # Calculate decile boundaries for length
    all_lengths = train_dataset['length']
    deciles = np.percentile(all_lengths, np.arange(0, 101, 10))
    
    # Add a combined stratification column (label + length decile)
    def add_strat_column(examples):
        strat_keys = []
        for label, length in zip(examples['label'], examples['length']):
            # Assign to decile bin (0-9)
            length_bin = np.digitize(length, deciles[1:-1])  # Exclude 0% and 100%
            # ClassLabel stores values as integers internally
            strat_key = f"{label}_{length_bin}"
            strat_keys.append(strat_key)
        examples['strat_key'] = strat_keys
        return examples
    
    train_dataset = train_dataset.map(add_strat_column, batched=True)
    
    # Convert strat_key to ClassLabel for stratification
    unique_strat_keys = sorted(set(train_dataset['strat_key']))
    train_dataset = train_dataset.cast_column(
        'strat_key',
        ClassLabel(names=unique_strat_keys)
    )
    
    # Perform stratified split
    split_dataset = train_dataset.train_test_split(
        test_size=validation_size,
        stratify_by_column='strat_key',
        seed=random_state
    )
    
    # Remove the temporary stratification column
    split_dataset['train'] = split_dataset['train'].remove_columns(['strat_key'])
    split_dataset['validation'] = split_dataset['test'].remove_columns(['strat_key'])
    
    # Create final DatasetDict with train, validation, and test
    processed_dataset = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['validation'],
        'test': dataset['test']
    })
    
    processed_dataset = processed_dataset.map(preprocess_function, batched=True)
    return processed_dataset


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


class CSVLoggingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_log.csv")
        self.initialized = False
        self.fieldnames = ["epoch", "step", "eval_loss", "eval_accuracy", "eval_recall", "eval_precision", "eval_f1"]
        self.buffer = {}
    
    def is_full(self):
        return all(key in self.buffer for key in self.fieldnames)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # Add step info
        logs = dict(logs)
        logs["step"] = state.global_step
        self.buffer.update(logs)
        
        if self.is_full():
            write_header = not os.path.exists(self.filepath)
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if write_header:
                    writer.writeheader()
                cleaned_buffer = {k: self.buffer[k] for k in self.fieldnames}
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