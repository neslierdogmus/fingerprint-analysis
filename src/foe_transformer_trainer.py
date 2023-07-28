from datasets import load_dataset
import evaluate
from transformers import DeiTImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, DefaultDataCollator
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import numpy as np
import torch
from PIL import Image


model_cp = "facebook/deit-base-distilled-patch16-224"
batch_size = 32

# TODO: dataset will be converted to patches of foe-dataset
dataset = load_dataset("keremberke/pokemon-classification", name = "full")
metric = evaluate.load("accuracy")

print(dataset.keys())

labels = dataset["train"].features["labels"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

image_processor = DeiTImageProcessor.from_pretrained(model_cp)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):

    """ example_batch["image"] = [Image.open(image_path).convert("RGB") for image_path in example_batch["image"]] """
    # Apply the transformations
    example_batch["pixel_values"] = [
        train_transforms(image) for image in example_batch["image"]
    ]
    # Convert the 'pixel_values' list to a tensor
    example_batch["pixel_values"] = torch.stack(example_batch["pixel_values"])
    return example_batch

def preprocess_val(example_batch):

    """ example_batch["image"] = [Image.open(image_path).convert("RGB") for image_path in example_batch["image"]] """
    # Apply the transformations
    example_batch["pixel_values"] = [val_transforms(image) for image in example_batch["image"]]
    # Convert the 'pixel_values' list to a tensor
    example_batch["pixel_values"] = torch.stack(example_batch["pixel_values"])
    return example_batch

dataset['train'].set_transform(preprocess_train)
dataset['validation'].set_transform(preprocess_val)


model = AutoModelForImageClassification.from_pretrained(
    model_cp,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True,
)

model_name = model_cp.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-pokemon",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

""" def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels} """

data_collator = DefaultDataCollator()

trainer = Trainer(
    model,
    args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


results = trainer.train()
print("kjgkjhgyj*****************-----------------***")
print(trainer.log())
""" print(results)
print(trainer.log_metrics('train', results.metrics)) """