import os
import csv
import json
from xml.parsers.expat import model
import torch
import evaluate
import numpy as np 
import os.path as osp
from sklearn.metrics import f1_score
from shutil import copyfile, rmtree
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_utils import preprocess
from torch.optim import AdamW
from huggingface_hub import HfApi, errors
from transformers import ElectraConfig, ElectraForSequenceClassification 
import copy
from data_utils import preprocess, clean_text, load_label_mapping

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
    
def initialize_student(teacher, mode="pruning", num_layers=6):
    """
    Initializes a student model based on the teacher.
    
    Args:
        teacher: The trained teacher model.
        mode: "pruning" (extract from teacher) or "st" (standard distillation from base).
        num_layers: Number of encoder layers for the student.
    """
    num_labels = teacher.config.num_labels
    
    # If pruning, we use teacher's specific config; if ST, we use the base model path
    base_path = teacher.config.name_or_path if mode == "pruning" else "google/electra-base-discriminator"
    
    student_config = ElectraConfig.from_pretrained(
        base_path,
        num_hidden_layers=num_layers,
        num_labels=num_labels,
    )

    student = ElectraForSequenceClassification(student_config)
    
    if mode == "st":
        print("Initializing Student from google/electra-base-discriminator (ST mode)...")
        source_model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator")
    else:
        print("Initializing Student by pruning the provided Teacher...")
        source_model = teacher
    student.electra.embeddings.load_state_dict(source_model.electra.embeddings.state_dict())

    # 5. Transfer Layer Weights (Skip-layer selection)
    pretrained_indices = [1, 3, 5, 7, 9, 11] 
    for student_idx, source_idx in enumerate(pretrained_indices):
        student.electra.encoder.layer[student_idx].load_state_dict(
            source_model.electra.encoder.layer[source_idx].state_dict()
        )
        
    if mode == "pruning":
        student.classifier.load_state_dict(source_model.classifier.state_dict())
    else:
        print("Classifier initialized randomly for ST mode.")

    return student

class CompressionTrainer:
    def __init__(self, loss_name, teacher, student, dataset, data_collator, tokenizer, weights, path="."):
        self.teacher = teacher
        self.student = student
        self.data_collator = data_collator
        self.train_dataset = dataset['train']
        self.train_dataset.set_format(type='torch')
        self.val_dataset = dataset['validation']
        self.val_dataset.set_format(type='torch')
        self.tokenizer = tokenizer
        self.loss_name = loss_name
        self.path = path
        LOSS_REGISTRY = {
            "ce": CompressionTrainer.pruning_loss,
            "kd": CompressionTrainer.student_teacher_loss,
            "hybrid": CompressionTrainer.hybrid_loss
        }
        self.chosen_loss = LOSS_REGISTRY[loss_name].__get__(self)
        self.weights = torch.from_numpy(weights)


    def inialize_training(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = 3 
        self.counter = 0 
        self.early_stop = False
        self.best_acc = 0.0
        self.num_epochs = 10
        self.batch_size = 32
        self.temperature = 2.0
        self.lr = 2e-5
        self.weight_decay=0.01
        self.warmup_steps=0
        self.alpha = 0.5  # weight for distillation loss
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def count_parameters(self, teacher):
        print(sum(p.numel() for p in teacher.parameters()))

    def pruning_loss(self, student_logits, teacher_logits, labels):
        loss = F.cross_entropy(student_logits, labels, weight=self.weights.to(self.device))
        return loss
    
    def student_teacher_loss(self, student_logits, teacher_logits, labels):
        # Soft loss (teacher guidance)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)
        return soft_loss
    
    def hybrid_loss(self, student_logits, teacher_logits, labels):
        hard_loss = self.pruning_loss(student_logits, teacher_logits, labels)
        soft_loss = self.student_teacher_loss(student_logits, teacher_logits, labels)
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss

    
    def train_model(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        # 2. Setup Scheduler
        num_training_steps = len(self.train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=num_training_steps
        )

        self.student.train()
        for epoch in range(self.num_epochs):
            if self.early_stop:
                print("Early stopping triggered. Training finished.")
                break
            total_loss = 0.0
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)


                # Teacher forward (no grad)
                with torch.no_grad():
                    if self.loss_name != "ce":
                        teacher_logits = self.teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        ).logits
                    else:
                        teacher_logits = None

                # Student forward
                student_logits =self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                ).logits

                # Distillation loss
                loss = self.chosen_loss(student_logits, teacher_logits, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            # print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_loss:.4f}")
            self.validate_model(self.val_dataloader, epoch)

    def validate_model(self, val_dataloader, epoch):
        self.student.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        # acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation Accuracy: {acc:.4f}")
        self.save_best_model(epoch, acc)

    def save_best_model(self, epoch, acc):
        if acc > self.best_acc:
            self.counter = 0
            self.best_acc = acc
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            self.student.save_pretrained(self.path)
            self.tokenizer.save_pretrained(self.path)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def iterative_with_trainer(teacher_model, processed_dataset, tokenizer, layers_to_drop, loss, data_collator, save_dir="students_iterative"): 
    """
    Iterative distillation using copy.deepcopy + layer popping with your Trainer class.

    teacher_model: HuggingFace teacher model (frozen)
    train_dataloader, val_dataloader: HuggingFace DataLoaders`
    tokenizer: tokenizer for saving/loading
    layers_to_drop: list of layers to drop in order (top layers first)
    num_epochs_per_student: number of epochs for each student
    save_dir: folder to save each student model
    """
    os.makedirs(save_dir, exist_ok=True)
    current_teacher = teacher_model

    for step, layer_idx in enumerate(layers_to_drop):
        print(f"\n--- Iteration {step}/{len(layers_to_drop)} ---")
        print(f"Dropping layer {layer_idx} for this student")

        # student imitate teacher
        student = copy.deepcopy(current_teacher)

        # Drop the specified layer
        student.electra.encoder.layer.pop(layer_idx)
        student.config.num_hidden_layers -= 1
        for param in student.parameters():
            param.requires_grad = True

        print("Teacher params:", count_parameters(current_teacher)[0])
        print("Student params:", count_parameters(student)[0])

        if step == len(layers_to_drop) - 1:
            step_save_path = os.path.join(save_dir, f"with_{loss}")
        else:
            step_save_path = os.path.join(save_dir, f"step{step}")

        trainer = CompressionTrainer(
            loss_name=loss,
            teacher=current_teacher, # Use the model from last step
            student=student,         # new student
            dataset=processed_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            weights=calc_class_weights(processed_dataset['train']),
            path=step_save_path
            )
    
        trainer.inialize_training()

        trainer.early_stop = False 
        
        trainer.train_model()
        
        current_teacher = copy.deepcopy(trainer.student).cpu().eval()


        print("\nIterative distillation finished!")
        print(f"Final student is saved in: {step_save_path}")
    return step_save_path  # folder of the final student

def iterative_with_trainer_same_teacher(teacher_model, processed_dataset, tokenizer, layers_to_drop, loss, data_collator, save_dir="students_iterative"): 
    """
    Iterative distillation using copy.deepcopy + layer popping with your Trainer class.

    teacher_model: HuggingFace teacher model (frozen)
    train_dataloader, val_dataloader: HuggingFace DataLoaders`
    tokenizer: tokenizer for saving/loading
    layers_to_drop: list of layers to drop in order (top layers first)
    num_epochs_per_student: number of epochs for each student
    save_dir: folder to save each student model
    """
    os.makedirs(save_dir, exist_ok=True)
    teacher = teacher_model
    current_student = copy.deepcopy(teacher)
    for step, layer_idx in enumerate(layers_to_drop):
        print(f"\n--- Iteration {step}/{len(layers_to_drop)} ---")
        print(f"Dropping layer {layer_idx} for this student")

        # student imitate teacher
        student = copy.deepcopy(current_student)

        # Drop the specified layer
        student.electra.encoder.layer.pop(layer_idx)
        student.config.num_hidden_layers -= 1
        for param in student.parameters():
            param.requires_grad = True

        print("Teacher params:", count_parameters(current_student)[0])
        print("Student params:", count_parameters(student)[0])

        if step == len(layers_to_drop) - 1:
            step_save_path = os.path.join(save_dir, f"{loss}")
        else:
            step_save_path = os.path.join(save_dir, f"{step}")

        trainer = CompressionTrainer(
            loss_name=loss,
            teacher=teacher, # Use the model from last step
            student=student,         # new student
            dataset=processed_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            weights=calc_class_weights(processed_dataset['train']),
            path=step_save_path
            )
    
        trainer.inialize_training()

        trainer.early_stop = False 
        
        trainer.train_model()
        
        current_student = copy.deepcopy(trainer.student).cpu().eval()


        print("\nIterative distillation finished!")
        print(f"Final student is saved in: {step_save_path}")
    return step_save_path  # folder of the final student
