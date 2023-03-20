import pandas as pd
import torch
import datasets
from tqdm import tqdm
import spacy
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-base"
source_lang = "de"
target_lang = "en"
prefix = "translate German to English: "
inverse_prefix = "translate English to German: "
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = spacy.load('en_core_web_sm')
de_parser = spacy.load("de_core_news_sm")
# Load a pre-trained tokenizer'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, max_length=300, truncation=True)
# Define the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def extract_paragraphs_from_file(data_path, labeled=True):
    """
    This function gets a path to the labeled/unlabeled data and extracts all the paragraphs from it into lists.
    :param data_path: The path to the labeled/unlabeled data
    :param labeled: Whether our file is labeled or not
    :return: 2 lists of paragraphs
    """
    # lists for German and English paragraphs, German paragraphs and English paragraphs
    split_paragraphs = []
    german_paragraphs = []
    english_paragraphs = []
    # Load the data from the file
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read()
    # Split the data into German and English paragraphs
    paragraphs = data.split("German:\n")[1:]
    # Iterate over the paragraphs
    for paragraph in paragraphs:
        if labeled:
            split_paragraphs += paragraph.split("English:\n")
    # In the unlabeled case
    if not labeled:
        split_paragraphs = paragraphs
    for i in range(0, len(split_paragraphs), 2):
        if labeled:
            german_paragraphs.append(split_paragraphs[i])
            english_paragraphs.append(split_paragraphs[i + 1][:-1])
        else:
            german_paragraphs.append(split_paragraphs[i][:-1])
            german_paragraphs.append(split_paragraphs[i + 1][:-1])
    return german_paragraphs, english_paragraphs


def paragraphs_to_dataset(german_paragraphs, english_paragraphs, labeled=True):
    """
    This function gets 2 lists of paragraphs in german and english and returns a dataset of them
    :param german_paragraphs: Paragraphs in German
    :param english_paragraphs: Paragraphs in English
    :param labeled: whether 'english_paragraphs' is being used
    :return: a dataset of the paragraphs
    """
    data = []
    for i in range(len(german_paragraphs)):
        '''
        # REMOVING LINES
        german_sentences, english_sentences = german_paragraphs[i].split('\n')[:-1], english_paragraphs[i].split('\n')[
                                                                                     :-1]
        german_sentences, english_sentences = ' '.join(str(e) for e in german_sentences), ' '.join(
            str(e) for e in english_sentences)
        '''
        if labeled:
            german_sentences, english_sentences = german_paragraphs[i][:-1], english_paragraphs[i][:-1]
            paragraph_dict = {'translation': {source_lang: german_sentences, target_lang: english_sentences}}
        else:
            german_sentences = german_paragraphs[i][:-1]
            paragraph_dict = {'translation': {source_lang: german_sentences, target_lang: ''}}
        # paragraph_dict = {'id': i, 'translation': {source_lang: german_sentences, target_lang: english_sentences}}
        data.append(paragraph_dict)
    # Create a Dataset object from the list of dictionaries
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset


def preprocess_function(examples):
    """
    This function Tokenizes every paragraph in the given batch with the tokenizer object of the transformer.
    :param examples: Batch of paragraphs from the dataset
    :return: The tokenized paragraphs
    """
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    # model_inputs = tokenizer(inputs, max_length=200, truncation=True)
    model_inputs = tokenizer(inputs, padding=True, max_length=300, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        # labels = tokenizer(targets, max_length=200, truncation=True)
        labels = tokenizer(targets, padding=True, max_length=300, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_hw1_files(file_path, tagged=True):
    """
    This function reads the files from homework1 and outputs the sentences from it
    :param file_path: A path to a file from HW1
    :param tagged: whether the file is labeled or not
    :return: list of the sentences
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            if tagged:
                sentence_words = []
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    sentence_words.append(cur_word)
                sentence = ' '.join(str(e) for e in sentence_words)
            else:
                sentence = line
            list_of_sentences.append(sentence)
    return list_of_sentences


def translate_to_german(sentence):
    """
    This function translates a single sentence from the target language to the source language
    with our pre-trained model and creates a dictionary object of the sentences.
    :param sentence: The sentence in the target language
    :return: A dictionary object of the sentences
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_text = inverse_prefix + sentence
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    paragraph_dict = {'translation': {source_lang: output_text, target_lang: sentence}}
    return paragraph_dict


def translate_sentences_to_german(list_of_sentences):
    """
    This function translate a list of sentences from the target language to the source language
    using a mapping of the function 'translate_to_german' and creates a corresponding dataset.
    :param list_of_sentences: List of sentences in the target language
    :return: A dataset of the sentences
    """
    data_series = pd.Series(list_of_sentences)
    data = map(translate_to_german, tqdm(data_series))
    # Create a Dataset object from the list of dictionaries
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset


def extract_from_unlabeled_paragraphs(paragraphs):
    """
    This function extracts from an unlabeled file the paragraph,
    roots and modifiers and returns 3 lists for all.
    :param paragraphs: List of paragraphs containing roots and modifiers
    as they appear in the file
    :return: A list for the paragraphs, a list for the roots and a list for the modifiers
    """
    raw_paragraphs = []
    roots = []
    modifiers = []
    for paragraph in paragraphs:
        split_paragraph = paragraph.split("Roots in English: ")
        raw_paragraphs.append(split_paragraph[0])
        split_parser = split_paragraph[1].split("Modifiers in English: ")
        roots.append(split_parser[0][:-1])
        modifiers.append(split_parser[1][:-1])
    return raw_paragraphs, roots, modifiers


def sentence_generator(paragraphs):
    for paragraph in paragraphs:
        yield paragraph
        # yield parser(paragraph)


def add_roots_and_modifiers(examples):
    """
    This function uses a pre-trained model for dependency parsing to find the roots and modifiers
    to the target sentence and adding them as a prefix to the source sentence
    :param examples: Batch of objects from the dataset
    :return: The batch of objects after update
    """
    # Paragraphs in english
    paragraphs = [example[target_lang] for example in examples["translation"]]
    de_paragraphs = [example[source_lang] for example in examples["translation"]]
    de_split_paragraphs = [par.split('\n') for par in de_paragraphs]
    # Using spacy's dependency parsing model on the paragraph
    docs = list(parser.pipe(sentence_generator(paragraphs), batch_size=1000))
    # Splitting the paragraph to multiple sentences
    sentences_list = [list(doc.sents) for doc in docs]
    idx_list = []
    for i in range(len(paragraphs)):
        if len(de_split_paragraphs[i]) == len(sentences_list[i]):
            idx_list += [i]
    for idx, sentences in enumerate(sentences_list):
        if idx in idx_list:
            de_par = de_split_paragraphs[idx]
            paragraph = ''
            # Finding the roots and modifiers
            for j, sentence in enumerate(sentences):
                roots_and_modifiers = '[Root: ' + str(sentence.root) + ' Modifiers: ' + \
                                      ', '.join(str(child) for child in sentence.root.children) + '] '
                paragraph += roots_and_modifiers
                paragraph += de_par[j]
            examples["translation"][idx][source_lang] = paragraph
        else:
            de_par = ' '.join(sen for sen in de_split_paragraphs[idx])
            de_par = re.split("(?<=[.!?]) +", de_par)
            if len(de_par) == len(sentences_list[idx]):
                paragraph = ''
                for j, sentence in enumerate(sentences):
                    roots_and_modifiers = '[Root: ' + str(sentence.root) + ' Modifiers: ' + \
                                          ', '.join(str(child) for child in sentence.root.children) + '] '
                    paragraph += roots_and_modifiers
                    paragraph += de_par[j]
                examples["translation"][idx][source_lang] = paragraph
    return examples


def add_roots_and_modifiers_unlabeled(paragraphs, roots, modifiers):
    """
    This function adds given roots and modifiers to the paragraphs as a prefix
    :param paragraphs: List of paragraphs
    :param roots: List of roots
    :param modifiers: List of modifiers
    :return: List of updated paragraphs
    """
    raw_paragraphs = paragraphs.copy()
    for i, paragraph in tqdm(enumerate(raw_paragraphs)):
        cur_roots = roots[i].split(',')
        cur_modifiers = modifiers[i].split('), (')
        cur_modifiers[0], cur_modifiers[-1] = cur_modifiers[0].replace('(', ''), cur_modifiers[-1].replace(')', '')
        sentences = raw_paragraphs[i].split('\n')
        if len(sentences) == len(cur_roots):
            for idx, sentence in enumerate(sentences):
                roots_and_modifiers = '[Root: ' + cur_roots[idx] + ' Modifiers: ' + cur_modifiers[idx] + ']'
                sentences[idx] = roots_and_modifiers + sentence
        raw_paragraphs[i] = '\n'.join(str(e) for e in sentences)
    return raw_paragraphs


def copy_sentences_to_new(parsed_tokenized_dataset, unparsed_tokenized_dataset):
    data = []
    for idx, p in tqdm(enumerate(parsed_tokenized_dataset)):
        # parsed_tokenized_dataset[idx]["translation"] = unparsed_tokenized_dataset[idx]["translation"]
        paragraph_dict = {'translation': {source_lang: unparsed_tokenized_dataset[idx]["translation"][source_lang],
                                          target_lang: parsed_tokenized_dataset[idx]["translation"][target_lang]}}
        data.append(paragraph_dict)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset


def copy_sentences(parsed_tokenized_dataset, unparsed_tokenized_dataset):
    for idx, p in tqdm(enumerate(parsed_tokenized_dataset)):
        parsed_tokenized_dataset[idx]["translation"] = unparsed_tokenized_dataset[idx]["translation"]
    return parsed_tokenized_dataset

'''
def copy_sentence(parsed_tokenized_sentence, unparsed_tokenized_sentence):
    parsed_tokenized_sentence["translation"] = unparsed_tokenized_sentence["translation"]
    return parsed_tokenized_sentence
    
 
def translate_list_to_german(list_of_sentences):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = []
    for sentence in tqdm(list_of_sentences):
        input_text = inverse_prefix + sentence
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        paragraph_dict = {'translation': {source_lang: output_text, target_lang: sentence}}
        data.append(paragraph_dict)
    # Create a Dataset object from the list of dictionaries
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset


def extracting3(path, labeled=True):
    """
  This function extract the sentences from only the paragraphs that have the same number of rows
  in the english paragraph and in the german paragraph.
  """
    split_paragraphs = []
    german_paragraphs = []
    english_paragraphs = []
    places = []
    s = 0
    num_lost_german = 0  # will count how many german sentences we throw
    num_lost_english = 0  # will count how many english sentences we throw
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    paragraphs = data.split("German:\n")[
                 1:]  # spliting the data for all the pair of paragraphs and remove the empy line at the end
    if not labeled:
        print(paragraphs)
    for paragraph in paragraphs:
        if labeled:
            split_paragraphs += paragraph.split("English:\n")  # split for englisg and german
    for i in range(0, len(split_paragraphs), 2):
        if labeled:
            ger = split_paragraphs[i].split('\n')[:-1]  # extracting the german sentences
            eng = split_paragraphs[i + 1].split('\n')[:-2]  # extracting the english sentences
            if len(eng) == len(ger):
                english_paragraphs += eng
                german_paragraphs += ger
            else:
                num_lost_german += len(ger)  # how many sentences in german we "throw"
                num_lost_english += len(eng)  # how many sentences in english we "throw"
    print(num_lost_german)
    print(num_lost_english)
    return german_paragraphs, english_paragraphs


def paragraphs_to_dataset3(german_paragraphs, english_paragraphs):
    """
    This function gets 2 lists of sentences in german and english and returns a dataset of them
    @return: a dataset of the sentences
    """
    data = []
    for i in range(len(german_paragraphs)):
        german_sentences, english_sentences = german_paragraphs[i].split('\n'), english_paragraphs[i].split('\n')
        german_sentences, english_sentences = ' '.join(str(e) for e in german_sentences), ' '.join(
            str(e) for e in english_sentences)
        paragraph_dict = {'translation': {source_lang: german_sentences, target_lang: english_sentences}}
        data.append(paragraph_dict)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return dataset
    
def add_roots_and_modifiers_temp(examples):
    """
    This function uses a pre-trained model for dependency parsing to find the roots and modifiers
    to the target sentence and adding them as a prefix to the source sentence
    :param examples: Batch of objects from the dataset
    :return: The batch of objects after update
    """
    paragraphs = [example[target_lang] for example in examples["translation"]]
    # Using spacy's dependency parsing model on the paragraph
    docs = list(parser.pipe(sentence_generator(paragraphs), batch_size=1000))
    # Splitting the paragraph to multiple sentences
    sentences_list = [list(doc.sents) for doc in docs]
    for idx, sentences in enumerate(sentences_list):
        # Finding the roots and modifiers
        roots_and_modifiers = 'Roots and Modifiers: '
        for sentence in sentences:
            if roots_and_modifiers != '':
                roots_and_modifiers += ', '
            roots_and_modifiers += str(sentence.root)
            roots_and_modifiers += ' (' + ', '.join(str(child) for child in sentence.root.children) + ')'
        roots_and_modifiers += '. '
        examples["translation"][idx][source_lang] = roots_and_modifiers + examples["translation"][idx][source_lang]
    return examples
    
    def add_roots_and_modifiers_unlabeled(raw_paragraphs, roots, modifiers):
    """
    This function adds given roots and modifiers to the paragraphs as a prefix
    :param raw_paragraphs: List of paragraphs
    :param roots: List of roots
    :param modifiers: List of modifiers
    :return: List of updated paragraphs
    """
    for i, paragraph in tqdm(enumerate(raw_paragraphs)):
        # TRYING AS A SUFFIX!
        # raw_paragraphs[i] = paragraph + 'Roots: ' + roots[i] + '. Modifiers: ' + modifiers[i] + '. '
        raw_paragraphs[i] = 'Roots: ' + roots[i] + '. Modifiers: ' + modifiers[i] + '. ' + paragraph
    return raw_paragraphs
    
    
def add_roots_and_modifiers(examples):
    """
    This function uses a pre-trained model for dependency parsing to find the roots and modifiers
    to the target sentence and adding them as a prefix to the source sentence
    :param examples: Batch of objects from the dataset
    :return: The batch of objects after update
    """
    paragraphs = [example[target_lang] for example in examples["translation"]]
    # Using spacy's dependency parsing model on the paragraph
    docs = list(parser.pipe(sentence_generator(paragraphs), batch_size=1000))
    # Splitting the paragraph to multiple sentences
    sentences_list = [list(doc.sents) for doc in docs]
    for idx, sentences in enumerate(sentences_list):
        # Finding the roots and modifiers
        roots = ''
        modifiers = ''
        for sentence in sentences:
            if roots != '':
                roots += ', '
            if modifiers != '':
                modifiers += ', '
            roots += str(sentence.root)
            modifiers += '(' + ', '.join(str(child) for child in sentence.root.children) + ')'
        sentence_prefix = 'Roots: ' + roots + '. Modifiers: ' + modifiers + '. '
        examples["translation"][idx][source_lang] = sentence_prefix + examples["translation"][idx][source_lang]
    return examples
'''
