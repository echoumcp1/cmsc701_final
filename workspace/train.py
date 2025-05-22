import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, IA3Config, TaskType
import torch.nn as nn

# 1) Load train & test CSVs
df_train = pd.read_csv("./genomic_species_subseq.csv")
df_test  = pd.read_csv("./genomic_species_subseq_test.csv")  # <-- your 400-row file
print(df_train.keys())
# 2) Build a shared ClassLabel on train genera only
labels = ClassLabel(names=sorted(df_train["species_epithet"].unique()))
df_train["label"] = labels.str2int(df_train["species_epithet"])
# map the *same* encoding onto test
df_test["label"]  = labels.str2int(df_test["species_epithet"])

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "test":  Dataset.from_pandas(df_test.reset_index(drop=True).loc[:399].reset_index(drop=True)),
})
# class-encode the HF “label” column
dataset = dataset.class_encode_column("label")


tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")
def tokenize_fn(batch):
    return tokenizer(
        batch["sequence"],
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )

dataset = dataset.rename_column("label", "labels")
tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)



model = AutoModelForSequenceClassification.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    num_labels=labels.num_classes
)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name)

peft_config = IA3Config(
    task_type=TaskType.SEQ_CLS,
    target_modules=[
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ],
    feedforward_modules=[
        "intermediate.dense",
        "output.dense",
    ]
)
model = get_peft_model(model, peft_config)

train_size   = len(tokenized["train"])
batch_size   = 32
steps_per_epoch = train_size // batch_size

training_args = TrainingArguments(
    output_dir="nt-genus-finetune",

    do_train=True,
    do_eval=True,

    # run eval once per epoch
    eval_steps=steps_per_epoch,

    # save checkpoint once per epoch
    save_steps=steps_per_epoch,
    save_total_limit=3,      

    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=50,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = (preds == labels).mean()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset= tokenized["test"],
    compute_metrics=compute_metrics,
    # label_names=["label"],
)


training_args.num_train_epochs = 1


trainer.args = training_args


num_epochs = 10   

for epoch in range(1, num_epochs+1):
    print(f"\n=== Starting epoch {epoch}/{num_epochs} ===")
    trainer.train()                       
    metrics = trainer.evaluate()           
    print(f"→ Test accuracy after epoch {epoch}: {metrics['eval_accuracy']:.4f}")

# 2) save the full fine-tuned model + tokenizer
save_dir = "nt-genus-finetune-final"
trainer.save_model(save_dir)                
tokenizer.save_pretrained(save_dir)        

print(f"Model and tokenizer saved to {save_dir}")