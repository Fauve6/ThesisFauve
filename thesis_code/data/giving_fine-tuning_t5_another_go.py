from transformers import AutoTokenizer
import jsonlines
import pandas as pd
import numpy as np
import nltk
import os
import csv
import itertools
import sys
#nltk.download('punkt')
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric


model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
prefix = 'translate question to SQL: '
max_input_length = 512
max_target_length = 64


def get_dict(file):
    question_line = []
    answer_line = []
    with jsonlines.open(file) as f:
        for line in f.iter():
            question_line.append(line['messages'][1]['content'])
            answer_line.append(line['messages'][2]['content'])
    my_dict = {"questions": question_line, "answers": answer_line}

    return my_dict


def get_data():
    train_dict = get_dict('train/for_t5/train_everything_english_without_val.jsonl')
    val_dict = get_dict('val/val_everything_english.jsonl')
    test_dict = get_dict('test/test_everything_en.jsonl')

    return train_dict, val_dict, test_dict


def preprocess_data(examples):
    inputs = [prefix + text for text in examples["questions"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answers"], max_length=max_target_length,
                           truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


if not os.path.isfile("dataset/dataset_train.csv"):
    train, val, test = get_data()
    with open('dataset/dataset_train.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        key_list = list(train.keys())
        limit = len(train['questions'])
        writer.writerow(train.keys())
        for i in range(limit):
            writer.writerow([train[x][i] for x in key_list])

    with open('dataset/dataset_val.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        key_list = list(val.keys())
        limit = len(val['questions'])
        writer.writerow(val.keys())
        for i in range(limit):
            writer.writerow([val[x][i] for x in key_list])

    with open('dataset/dataset_test.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        key_list = list(test.keys())
        limit = len(test['questions'])
        writer.writerow(test.keys())
        for i in range(limit):
            writer.writerow([test[x][i] for x in key_list])

dataset = load_dataset("csv", data_dir='dataset', sep=",")
tokenized_datasets = dataset.map(preprocess_data, batched=True)

batch_size = 8
model_name = "t5-base"
model_dir = f"model/{model_name}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=25,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

trainer.save_model("model/experiments/t5_model_finetuned_everything_en")

