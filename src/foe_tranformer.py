from datasets import load_dataset
import evaluate

model_cp = "facebook/deit-base-distilled-patch16-224"
batch_size = 32

dataset = load_dataset("keremberke/pokemon-classification", name = "full")
metric = evaluate.load("accuracy")

labels = dataset["train"].features["labels"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

