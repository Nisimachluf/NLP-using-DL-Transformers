import os
import os.path as osp
import json
import optuna
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback
from utils import load_hf_classifier, get_metrics_fn, preprocess, calc_class_weights, WeightedTrainer

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 15),
        # "per_device_train_batch_size": trial.suggest_categorical(
            # "per_device_train_batch_size", [8, 16, 32, 64]
        # ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.02),
    }

def compute_objective(metrics):
    return metrics["eval_f1"]   # or eval_loss (minimize)

def get_model_init(model_name):
    def model_init():
        return load_hf_classifier(model_name, 6, freeze_backbone=False)[0]
    return model_init

def write_hparams(model_name, best_run):
    hparams_dir = "hparams"
    hparams_file = f"{hparams_dir}/{model_name}.json"
    os.makedirs(osp.dirname(hparams_file), exist_ok=True)
    print(f"Writing best hyperparameters for {model_name} to {hparams_file}")
    with open(hparams_file, "w") as f:
        json.dump(best_run.hyperparameters, f)
    return

def load_hparams(model_name):
    hparams_dir = "hparams"
    hparams_file = f"{hparams_dir}/{model_name}.json"
    if not osp.exists(hparams_file):
        return None
    print(f"Loading best hyperparameters for {model_name} from {hparams_file}")
    with open(hparams_file, "r") as f:
        hparams = json.load(f)
    return hparams

def tune_hyperparams(model_name, dataset, weighted=False):

    hparams = load_hparams(model_name)
    if hparams is not None:
        return hparams
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model, tokenizer = load_hf_classifier(model_name, 6, freeze_backbone=False)
    dataset = preprocess(dataset, tokenizer)
    
    training_args = TrainingArguments(
#         output_dir=output_dir,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        push_to_hub=False,
    )


    trainer_args = {
    "model_init": get_model_init(model_name),
    "args": training_args,
    "train_dataset": dataset["train"],
    "eval_dataset": dataset["validation"],
    "processing_class": tokenizer,
    "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    "compute_metrics": get_metrics_fn(),
    "callbacks": [EarlyStoppingCallback(early_stopping_patience=2,
                                        early_stopping_threshold=0.0)
                  ],
    }
    if weighted:
        trainer_args["class_weights"] = calc_class_weights(dataset["train"])
        TrainerClass = WeightedTrainer
    else:
        TrainerClass = Trainer
        
    trainer = TrainerClass(**trainer_args)
    
    
    best_run = trainer.hyperparameter_search(
        backend="optuna",
        direction="maximize",
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=30,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
   
    write_hparams(model_name, best_run)
    return best_run.hyperparameters