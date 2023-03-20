import numpy as np
import pandas as pd
import evaluate
from transformers import AutoTokenizer
import torch
import datasets
from tqdm import tqdm
import re

model_name = "t5-base"
# Load a pre-trained tokenizer'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, max_length=300, truncation=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
source_lang = "de"
target_lang = "en"
prefix = "translate German to English: "


# Evaluation functions- as given
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def read_file(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def compute_metrics(tagged_en, true_en):
    metric = evaluate.load("sacrebleu")
    # metric = evaluate.load("accuracy")
    tagged_en = [x.strip().lower() for x in tagged_en]
    true_en = [x.strip().lower() for x in true_en]

    result = metric.compute(predictions=tagged_en, references=true_en)
    result = result['score']
    result = round(result, 2)
    return result


def calculate_score(file_path1, file_path2):
    file1_en, file1_de = read_file(file_path1)
    file2_en, file2_de = read_file(file_path2)
    print(len(file1_de), len(file2_de))
    print(len(file1_en), len(file2_en))
    print(file1_en[0], file2_en[0])
    print(file1_en[-1], file2_en[-1])
    for sen1, sen2 in zip(file1_de, file2_de):
        if sen1.strip().lower() != sen2.strip().lower():
            raise ValueError('Different Sentences')
    score = compute_metrics(file1_en, file2_en)
    print(score)


def calc_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = compute_metrics(decoded_preds, decoded_labels)
    result = {"bleu": result}
    return result


def translate_dataset(model, unlabeled_dataset):
    """
    This function translate the paragraphs from the source to target language
    and updates and returns the dataset after prediction.
    :param model: The pre-trained model
    :param unlabeled_dataset: The dataset before predictions
    :return: The dataset after predictions
    """
    data = []
    for i, par in tqdm(enumerate(unlabeled_dataset)):
        input_par = par["translation"][source_lang]
        input_par = prefix + input_par
        input_ids = tokenizer.encode(input_par, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=400, num_beams=5)
        output_par = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if output_par.isspace():
            output_par = input_par
        # Might help a little with the output format
        # output_par = re.split("(?<=[.!?]) +", output_par)
        # output_par = '\n'.join(s for s in output_par)

        paragraph_dict = {'translation': {source_lang: unlabeled_dataset[i]["translation"][source_lang],
                                          target_lang: output_par}}
        data.append(paragraph_dict)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset


def write_to_file(dataset, path):
    text = ""
    for par in tqdm(dataset):
        text += "German:\n"
        text += par["translation"][source_lang] + '\n'
        text += "English:\n"
        # text += par["translation"][target_lang] + '\n\n'
        p = par["translation"][target_lang]
        if p == '\n':
            p = '.'
        text += p + '\n\n'
    f = open(path, "w")
    f.write(text)
    f.close()


def evaluate_model(model, parsed_val_dataset):
    tagged_en, true_en = [], []
    # tagged_en = parsed_val_dataset.map(translate_to_target, model, batched=True)
    for i, par in tqdm(enumerate(parsed_val_dataset)):
        input_text = par["translation"][source_lang]
        input_text = prefix + input_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=400, num_beams=5)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Might help a little with the output format
        # output_text = re.split("(?<=[.!?]) +", output_text)
        # output_text = '\n'.join(s for s in output_text)
        tagged_en.append(output_text)
        true_en.append(par["translation"][target_lang])
        if i > 50:
           break
    result = compute_metrics(tagged_en, true_en)
    print(result)
    print(tagged_en[0])
    print(" ----- ")
    print(true_en[0])
    return result


'''
    
    
def translate_to_target(examples, model):
    input_text = [example[source_lang] for example in examples["translation"]]
    print(input_text)
    # input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_ids = tokenizer(input_text, padding=True, max_length=300, truncation=True, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, max_length=400, num_beams=6)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # paragraph_dict = {'translation': {source_lang: output_text, target_lang: sentence}}
    return output_text
'''

