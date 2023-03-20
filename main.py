import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, load_metric, concatenate_datasets

from preprocessing import *
from evaluation import *

# Data paths
TRAIN_PATH_LABELED = './data/train.labeled'
VAL_PATH_LABELED = './data/val.labeled'

VAL_PATH_UNLABELED = './data/val.unlabeled'
COMP_PATH_UNLABELED = './data/comp.unlabeled'

HW1_TRAIN1_PATH = './data/hw1_train1.wtag'
HW1_TRAIN2_PATH = './data/hw1_train2.wtag'
HW1_TEST1_PATH = './data/hw1_test1.wtag'
HW1_COMP1_PATH = './data/hw1_comp1.words'
HW1_COMP2_PATH = './data/hw1_comp2.words'

OUT_PATH = './results'
SAVED_PATH = './saved'
model_name = "t5-base"
source_lang = "de"
target_lang = "en"
prefix = "translate German to English: "
inverse_prefix = "translate English to German: "
metric = load_metric("sacrebleu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())

    # Extract paragraphs from train and validation files
    german_paragraphs_train, english_paragraphs_train = extract_paragraphs_from_file(TRAIN_PATH_LABELED)
    german_paragraphs_val, english_paragraphs_val = extract_paragraphs_from_file(VAL_PATH_LABELED)
    paragraphs_val_unlabeled, _ = extract_paragraphs_from_file(VAL_PATH_UNLABELED, False)

    val_raw_paragraphs, val_roots, val_modifiers = extract_from_unlabeled_paragraphs(paragraphs_val_unlabeled)
    parsed_german_paragraphs_val = add_roots_and_modifiers_unlabeled(val_raw_paragraphs, val_roots, val_modifiers)

    # Create a dataset from the paragraphs
    train_dataset_unparsed = paragraphs_to_dataset(german_paragraphs_train, english_paragraphs_train)
    val_dataset = paragraphs_to_dataset(parsed_german_paragraphs_val, english_paragraphs_val)
    val_dataset_unparsed = paragraphs_to_dataset(german_paragraphs_val, english_paragraphs_val)

    # Adding roots and modifiers in English as a prefix to each paragraph in German
    train_dataset = train_dataset_unparsed.map(add_roots_and_modifiers, batched=True)
    print(train_dataset, val_dataset)

    # Load a pre-trained tokenizer'
    # , padding='max_length', max_length=200
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, max_length=300, truncation=True)
    # Define the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    '''
    Create a data collator- "DataCollatorForSeq2Seq" takes in a list of examples, where each example consists of a
    source sequence and a target sequence, and collates them into batches that can be fed into a seq2seq model for
    training or inference. It handles tasks such as padding the sequences to a fixed length, truncating sequences that
    exceed the maximum length, and creating attention masks to indicate which tokens are padding tokens and which are real tokens.
    '''
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # processing and adding HW1 data
    hw1_train1_sentences = preprocess_hw1_files(HW1_TRAIN1_PATH, True)
    hw1_comp1_sentences = preprocess_hw1_files(HW1_COMP1_PATH, False)
    hw1_train2_sentences = preprocess_hw1_files(HW1_TRAIN2_PATH, True)
    hw1_comp2_sentences = preprocess_hw1_files(HW1_COMP2_PATH, False)
    hw1_test1_sentences = preprocess_hw1_files(HW1_TEST1_PATH, True)

    hw1_sentences = hw1_train1_sentences + hw1_comp1_sentences + hw1_train2_sentences + hw1_comp2_sentences \
                    + hw1_test1_sentences
    print("Translating HW1 data...")
    hw1_dataset_unparsed = translate_sentences_to_german(hw1_sentences)
    hw1_dataset = hw1_dataset_unparsed.map(add_roots_and_modifiers, batched=True)
    train_dataset = concatenate_datasets([hw1_dataset, train_dataset])
    train_dataset_unparsed = concatenate_datasets([hw1_dataset_unparsed, train_dataset_unparsed])

    # Tokenize the dataset
    # WITHOUT DEPENDENCY PARSING
    # train_tokenized_dataset = train_dataset_unparsed.map(preprocess_function, batched=True)
    # val_tokenized_dataset = val_dataset_unparsed.map(preprocess_function, batched=True)
    # WITH DEPENDENCY PARSING
    train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
    val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)

    train_tokenized_dataset = copy_sentences(train_tokenized_dataset, train_dataset_unparsed)
    val_tokenized_dataset = copy_sentences(val_tokenized_dataset, val_dataset_unparsed)

    train_tokenized_dataset.set_format('torch')
    val_tokenized_dataset.set_format('torch')

    print(train_tokenized_dataset, val_tokenized_dataset)
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Defining the training arguments and the trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUT_PATH,
        evaluation_strategy="epoch",
        # learning_rate=2e-5,  # default 5e-05
        per_device_train_batch_size=4,  # default 8
        per_device_eval_batch_size=4,  # default 8
        weight_decay=0.01,  # default 0.0
        save_total_limit=3,  # default None
        num_train_epochs=3,  # default 3.0
        predict_with_generate=True,
        # Whether to use generate to calculate generative metrics (ROUGE, BLEU), default False
        fp16=True,  # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training, default False
        generation_max_length=400,
        generation_num_beams=5,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=calc_metrics,
    )

    trainer.train()
    trainer.save_model(SAVED_PATH)
