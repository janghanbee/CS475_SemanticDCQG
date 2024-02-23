import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from config import *
# from common.constants import NLP
from util.file_utils import load
import re
import random
import codecs
import json

def get_match_spans(pattern, input):
    spans = []
    for match in re.finditer(re.escape(pattern.lower()), input.lower()):
        spans.append(match.span())
    return spans

def normalize_text(text):
    text = text.replace("''", '" ').replace("``", '" ')
    return text.strip()


def get_processed_hotpot_baseline_examples(filename, debug=False, debug_length=20, shuffle=False):
    """
    Get a list of raw examples given input SQuAD1.1-Zhou filename.
    """
    print("Start get HotpotQA raw examples ... ")
    start = datetime.now()
    examples = []
    with codecs.open(filename, encoding="utf-8") as fh:
        data = json.load(fh)
        num_examples = 0
        for item in tqdm(data):
            context = dict(item['context']) #context
            ans_sent = " ".join([context[name][idx].strip() for name, idx in item['supporting_facts']]) #supporting fact가 포함된 문장만 (supporting fact: [뒷받침 되는 키워드, 해당 키워드가 context에서 몇번 째 문장에 있는지])
            answer_text = item['answer']  #정답
            question = item['question'] #질문

            answer_spans = get_match_spans(answer_text, ans_sent) #ans_sent(정답을 맞히기 위해 필요한 문장들) 에서 정답의 index
            if len(answer_spans) == 0:
                continue
            answer_start = answer_spans[0][0]

            example = {
                "paragraph": normalize_text(ans_sent), #문장들
                "question": normalize_text(question), 
                "answer": normalize_text(answer_text), 
                "answer_start": answer_start, 
                "para_id": num_examples,  #data 인덱싱
                "inst_id": item['_id'] #data 내에 포함된 아이디
            }
            examples.append(example)
            num_examples += 1
            if debug and num_examples >= debug_length:
                break
        if shuffle:
            random.shuffle(examples)
    print(("Time of get raw examples: {}").format(datetime.now() - start))
    print("Number of raw examples: ", len(examples))
    return examples

def get_augmented_sents_examples(augmented_sentences_pkl_file, debug=False, debug_length=20, sent_limit=100, ans_limit=30):
    """
    This is used to load the augmented sentences data that generated by DA_main.py
    """
    examples = load(augmented_sentences_pkl_file)
    result = []

    para_id = 0
    for example in tqdm(examples):
        ans_sent = example["context"]

        for info in example["selected_infos"]:
            answer_text = info["answer"]["answer_text"]
            answer_start = info["answer"]["char_start"]
            # filter
            answer_bio_ids = info["answer"]["answer_bio_ids"]
            answer_length = answer_bio_ids.count("B") + answer_bio_ids.count("I")
            if (len(example["ans_sent_doc"]) > sent_limit or answer_length > ans_limit):
                continue
            for clue in info["clues"]:
                clue_text = clue["clue_text"]
                clue_start = ans_sent.find(clue_text)
                if clue_start < 0:  # not -1
                    continue

                for style_text in info["styles"]:
                    output_e = {
                        "paragraph": ans_sent,

                        "question": "",
                        "ques_type": style_text,  # string type

                        "answer": answer_text,
                        "answer_start": answer_start,

                        "clue": clue_text,
                        "clue_start": clue_start,
                        "para_id": example["sid"]}  # because our paragraph is sentence actually.
                    result.append(output_e)
        para_id += 1
        if debug and para_id >= debug_length:
            break
    return result


def get_dataset(tokenizer, dataset_cache, path, split='train', filetype='baseline', debug=False, debug_length=20):
    # Load question data
    if dataset_cache and os.path.isfile(dataset_cache):  #!!! NOTICE: make sure dataset_cache is correct version.
        print("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        return data

    data = get_positional_dataset_from_file(tokenizer, file=path, filetype=filetype, debug=debug, debug_length=debug_length)

    if dataset_cache:
        torch.save(data, dataset_cache)

    print("Dataset cached at %s", dataset_cache)

    return data


def get_position(para_ids, ans_ids, ans_prefix_ids):
    diff_index = -1
    # Find the first token where the paragraph and answer prefix differ
    for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
        if pid != apid:
            diff_index = i
            break
    if diff_index == -1:
        diff_index = min(len(ans_prefix_ids), len(para_ids))
    # Starting from this token, we take a conservative overlap
    return (diff_index, min(diff_index + len(ans_ids), len(para_ids)))


def get_positional_dataset_from_file(tokenizer, file, filetype="baseline", debug=False, debug_length=20):
    if filetype == "hotpot_sub" or filetype == "hotpot_comp":
        return get_positional_hotpot_dataset_from_file(tokenizer, file, filetype, debug=debug, debug_length=debug_length)
    elif filetype == "baseline":
        return get_positional_hotpot_baseline_dataset_from_file(tokenizer, file, debug=debug, debug_length=debug_length)
    elif filetype == "augmented_sents":
        data = get_augmented_sents_examples(file, debug, debug_length)
    # elif filetype == "squad":  
        # data = get_raw_examples(file, filetype, debug, debug_length)  # NOTICE: add handler for data augmented input data.
        # data = get_processed_examples(data, debug)
        
    truncated_sequences = 0
    for inst in tqdm(data):
        inst['answer_position'] = inst['answer_start']
        clue_exist = (inst['clue_start'] is not None)
        if clue_exist:
            inst['clue_position'] = inst['clue_start']

        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position']])

        if clue_exist:
            tokenized_clue = tokenizer.tokenize(inst['clue'])
            tokenized_clue_prefix = tokenizer.tokenize(inst['paragraph'][:inst['clue_position']])
        else:
            tokenized_clue = []

        tokenized_qtype = tokenizer.tokenize(inst['ques_type'])

        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + len(tokenized_clue) + len(tokenized_qtype) + 6

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + len(tokenized_clue) + len(tokenized_qtype) + 6 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
        inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)

        if clue_exist:
            inst['clue'] = tokenizer.convert_tokens_to_ids(tokenized_clue)
            clue_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_clue_prefix)
            inst['clue_position_tokenized'] = get_position(inst['paragraph'], inst['clue'], clue_prefix_ids)

        inst['style'] = tokenizer.convert_tokens_to_ids(tokenized_qtype)
        pass

    print("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data

def get_positional_hotpot_dataset_from_file(tokenizer, file, filetype, debug=False, debug_length=20):
    data = get_processed_hotpot_examples(file, filetype, debug)

    truncated_sequences = 0
    for inst in tqdm(data):
        inst['answer_position'] = inst['answer_start']

        tokenized_sup = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_sub = tokenizer.tokenize(inst['subj'])
        tokenized_verb = tokenizer.tokenize(inst['verb'])
        tokenized_obj = tokenizer.tokenize(inst['obj'])
        tokenized_prep = tokenizer.tokenize(inst['prep'])


        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_start']]) if inst['answer_start'] is not None else None
        tokenized_subj_prefix = tokenizer.tokenize(inst['subj'][:inst['answer_in_subj_start']]) if inst['answer_in_subj_start'] is not None else None
        tokenized_obj_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_in_obj_start']]) if inst['answer_in_obj_start'] is not None else None
        tokenized_prep_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_in_prep_start']]) if inst['answer_in_prep_start'] is not None else None

        if filetype == 'hotpot_comp':
            tokenized_question1 = tokenizer.tokenize(inst['question1'])
            total_seq_len = len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question1) + len(tokenized_question) + 10
        else:
            total_seq_len = len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question) + 9

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_sup = tokenized_sup[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            if filetype == 'hotpot_comp':
                assert len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question1) + len(tokenized_question) + 10 < tokenizer.max_len
            else:
                assert len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question) + 9 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_sup)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        inst['subj'] = tokenizer.convert_tokens_to_ids(tokenized_sub)
        inst['verb'] = tokenizer.convert_tokens_to_ids(tokenized_verb)
        inst['obj'] = tokenizer.convert_tokens_to_ids(tokenized_obj)
        inst['prep'] = tokenizer.convert_tokens_to_ids(tokenized_prep)
        if filetype == 'hotpot_comp':
            inst['question1'] = tokenizer.convert_tokens_to_ids(tokenized_question1)

        inst['answer_position_tokenized'] = None
        inst['answer_position_in_subj_tokenized'] = None
        inst['answer_position_in_obj_tokenized'] = None
        inst['answer_position_in_prep_tokenized'] = None
        if inst['answer_start'] is not None:
            ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
            inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)
        if inst['answer_in_subj_start'] is not None:
            ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_subj_prefix)
            inst['answer_position_in_subj_tokenized'] = get_position(inst['subj'], inst['answer'], ans_prefix_ids)
        if inst['answer_in_obj_start'] is not None:
            ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_obj_prefix)
            inst['answer_position_in_obj_tokenized'] = get_position(inst['obj'], inst['answer'], ans_prefix_ids)
        if inst['answer_in_prep_start'] is not None:
            ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_prep_prefix)
            inst['answer_position_in_prep_tokenized'] = get_position(inst['prep'], inst['answer'], ans_prefix_ids)

        pass

    print("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data

# def get_positional_hotpot_dataset_from_file_backup(tokenizer, file, filetype, debug=False, debug_length=20):
#     data = get_processed_hotpot_examples(file, filetype, debug)
#
#     truncated_sequences = 0
#     for inst in tqdm(data):
#         inst['answer_position_in_sup'] = inst['answer_start']
#         inst['answer_position_in_rel'] = inst['answer_in_rel_start']
#
#         tokenized_sup = tokenizer.tokenize(inst['paragraph'])
#         tokenized_question = tokenizer.tokenize(inst['question'])
#         tokenized_answer = tokenizer.tokenize(inst['answer'])
#         tokenized_sub = tokenizer.tokenize(inst['subj'])
#         tokenized_verb = tokenizer.tokenize(inst['verb'])
#         tokenized_obj = tokenizer.tokenize(inst['obj'])
#         tokenized_prep = tokenizer.tokenize(inst['prep'])
#
#         tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position_in_sup']])
#         tokenized_rel_prefix = tokenizer.tokenize(inst['subj'][:inst['answer_position_in_rel']]) if inst["ans_in_sub"] else tokenizer.tokenize(inst['obj'][:inst['answer_position_in_rel']])
#
#         if filetype == 'hotpot_comp':
#             tokenized_question1 = tokenizer.tokenize(inst['question1'])
#             total_seq_len = len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question1) + len(tokenized_question) + 10
#         else:
#             total_seq_len = len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question1) + 9
#
#         if total_seq_len > tokenizer.max_len:
#             # Heuristic to chop off extra tokens in paragraphs
#             tokenized_sup = tokenized_sup[:-1 * (total_seq_len - tokenizer.max_len + 1)]
#             truncated_sequences += 1
#             if filetype == 'hotpot_comp':
#                 assert len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question1) + len(tokenized_question) + 10 < tokenizer.max_len
#             else:
#                 assert len(tokenized_sup) + len(tokenized_sub) + len(tokenized_verb) + len(tokenized_obj) + len(tokenized_prep) + len(tokenized_answer) + len(tokenized_question) + 9 < tokenizer.max_len
#
#         inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_sup)
#         inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
#         inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
#         inst['subj'] = tokenizer.convert_tokens_to_ids(tokenized_sub)
#         inst['verb'] = tokenizer.convert_tokens_to_ids(tokenized_verb)
#         inst['obj'] = tokenizer.convert_tokens_to_ids(tokenized_obj)
#         inst['prep'] = tokenizer.convert_tokens_to_ids(tokenized_prep)
#         if filetype == 'hotpot_comp':
#             inst['question1'] = tokenizer.convert_tokens_to_ids(tokenized_question1)
#         ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
#         rel_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_rel_prefix)
#         inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)
#         inst['answer_position_in_rel_tokenized'] = get_position(inst['subj'], inst['answer'], rel_prefix_ids) if inst["ans_in_sub"] else get_position(inst['obj'], inst['answer'], rel_prefix_ids)
#
#         pass
#
#     print("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))
#
#     return data

def get_positional_hotpot_baseline_dataset_from_file(tokenizer, file, debug=False, debug_length=20):
    data = get_processed_hotpot_baseline_examples(file, debug)

    truncated_sequences = 0
    for inst in tqdm(data):
        inst['answer_position'] = inst['answer_start']

        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position']])

        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
        inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)

        pass

    print("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data

# examples = get_raw_examples("../../../../../Datasets/original/SQuAD1.1-Zhou/train.txt", filetype="squad", debug=True, debug_length=20)
# examples = get_processed_examples(examples, debug=True)
