from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from preprocessing import *
from evaluation import *

VAL_PATH_LABELED = './data/val.labeled'
VAL_PATH_UNLABELED = './data/val.unlabeled'
COMP_PATH_UNLABELED = './data/comp.unlabeled'
model_name = "t5-base"
VAL_PRED_PATH = 'val_214169377_213496110.labeled'
COMP_PRED_PATH = 'comp_214169377_213496110.labeled'
# LAST_CHECKPOINT_PATH = './results/checkpoint-22500'
LAST_CHECKPOINT_PATH = './saved'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, max_length=300, truncation=True)
model = AutoModelForSeq2SeqLM.from_pretrained(LAST_CHECKPOINT_PATH).to(device)
# Pre-process comp file
paragraphs_comp_unlabeled, _ = extract_paragraphs_from_file(VAL_PATH_UNLABELED, False)
comp_raw_paragraphs, comp_roots, comp_modifiers = extract_from_unlabeled_paragraphs(paragraphs_comp_unlabeled)
parsed_german_paragraphs_comp = add_roots_and_modifiers_unlabeled(comp_raw_paragraphs, comp_roots, comp_modifiers)
# Pre-process validation file
german_paragraphs_val, english_paragraphs_val = extract_paragraphs_from_file(VAL_PATH_LABELED)
paragraphs_val_unlabeled, _ = extract_paragraphs_from_file(VAL_PATH_UNLABELED, False)
val_raw_paragraphs, val_roots, val_modifiers = extract_from_unlabeled_paragraphs(paragraphs_val_unlabeled)
parsed_german_paragraphs_val = add_roots_and_modifiers_unlabeled(val_raw_paragraphs, val_roots, val_modifiers)

# Create datasets of the files
val_dataset = paragraphs_to_dataset(parsed_german_paragraphs_val, english_paragraphs_val)
val_dataset_unlabeled = paragraphs_to_dataset(parsed_german_paragraphs_val, '', False)
comp_dataset_unlabeled = paragraphs_to_dataset(parsed_german_paragraphs_comp, '', False)
val_dataset_unparsed = paragraphs_to_dataset(val_raw_paragraphs, '', False)
comp_dataset_unparsed = paragraphs_to_dataset(comp_raw_paragraphs, '', False)

# Prediction
# val_dataset_labeled = translate_dataset(model, val_dataset_unlabeled)
# comp_dataset_labeled = translate_dataset(model, comp_dataset_unlabeled)

# Removing the roots and modifiers before writing the predictions
# val_dataset_labeled = copy_sentences_to_new(val_dataset_labeled, val_dataset_unparsed)
# comp_dataset_labeled = copy_sentences_to_new(comp_dataset_labeled, comp_dataset_unparsed)

evaluate_model(model, val_dataset)

# Write the results into files
# write_to_file(val_dataset_labeled, VAL_PRED_PATH)
# write_to_file(comp_dataset_labeled, COMP_PRED_PATH)

calculate_score(VAL_PRED_PATH, VAL_PATH_LABELED)
