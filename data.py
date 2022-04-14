import random
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np


def iter_valid_records_cat(model, dataset, run, batch_size, tokenizer):
    batch = {'query_id': [], 'doc_id': [], 'pairs_q_doc': []}
    for qid, did, q_text, doc_text in _iter_valid_records_cat(model, dataset, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['pairs_q_doc'].append((q_text, doc_text))
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship_cat(batch, tokenizer)
            batch = {'query_id': [], 'doc_id': [], 'pairs_q_doc': []}
    if len(batch['query_id']) > 0:
        yield _pack_n_ship_cat(batch, tokenizer)

def _iter_valid_records_cat(model, dataset, run):
    ds_queries, ds_docs = dataset
    for qid in run:
        q_text = ds_queries.get(qid)
        if q_text is None:
            tqdm.write(f'missing query {qid}! Skipping')
            continue            
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            yield qid, did, q_text, doc
            
def _pack_n_ship_cat(batch, tokenizer):
    side_input = tokenizer(list(np.array(batch['pairs_q_doc']).reshape(1,-1)[0]),
                                  add_special_tokens=True, 
                                  max_length=512,
                                  truncation= True,
                                  return_overflowing_tokens=False,
                                  return_special_tokens_mask=False,
                                  return_token_type_ids=True,
                                  padding='max_length',
                                  return_tensors="pt",
                                  )

    model_input = tokenizer.batch_encode_plus(batch['pairs_q_doc'],
                                              add_special_tokens=True,
                                              max_length=512,
                                              truncation= True,
                                              return_overflowing_tokens=False,
                                              return_special_tokens_mask=False,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_tensors="pt",
                                              )



    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'model_input': model_input,
        'side_input': side_input
    }

def iter_train_pairs_cat(model, dataset, train_pairs, qrels, batch_size, tokenizer):
    batch = {'query_id': [], 'doc_id': [], 'pairs_q_doc': []}
    for qid, did, q_text, doc_text in _iter_train_pairs_cat(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['pairs_q_doc'].append((q_text, doc_text))
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship_cat(batch, tokenizer)
            batch = {'query_id': [], 'doc_id': [], 'pairs_q_doc': []}

def _iter_train_pairs_cat(model, dataset, train_pairs, qrels):
    ds_queries, ds_docs = dataset
    qids = list(train_pairs.keys())
    random.shuffle(qids) 
    for qid in qids:
        pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
        if len(pos_ids) == 0:
            tqdm.write("no positive labels for query %s " % qid)
            continue
        pos_ids_all = set(pos_ids)
        for pos_id in pos_ids:
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_all]
            if len(neg_ids) == 0:
                tqdm.write("no negative labels for query %s " % qid)
                continue
            neg_id = random.choice(neg_ids)
            q_text = ds_queries.get(qid)
            if q_text is None:
                tqdm.write(f'missing query {qid}! Skipping')
                continue                   
            
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            if neg_doc is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            yield qid, pos_id, q_text, pos_doc
            yield qid, neg_id, q_text, neg_doc


def read_datafiles(files):
    queries = {}
    docs = {}
    for file in files:
        tmp_df = pd.read_csv(file, names=['type', 'id', 'text'], sep='\t', dtype={'type':str , 'id':str, 'text':str})
        for line in tqdm(tmp_df.itertuples(), desc='loading datafile (by line)', leave=False):
            cols = [line.type, line.id, line.text]
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[str(c_id)] = c_text 
            if c_type == 'doc':
                docs[str(c_id)] = c_text 
    print("{} queries and {} docs are reazd".format(len(queries), len(docs)))
    return queries, docs


def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        if len(line.split())<6:
            print(line)
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result


def iter_train_pairs(model, dataset, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_train_pairs(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}



def _iter_train_pairs(model, dataset, train_pairs, qrels):
    ds_queries, ds_docs = dataset
    missing_doc_count = set()
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                tqdm.write("no positive labels for query %s " % qid)
                continue
            pos_id = random.choice(pos_ids)
            pos_ids_lookup = set(pos_ids)
            pos_ids = set(pos_ids)
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                tqdm.write("no negative labels for query %s " % qid) 
                continue
            neg_id = random.choice(neg_ids)


            query_tok = model.tokenize(ds_queries[qid])
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                missing_doc_count.add(pos_doc)
                continue
            if neg_doc is None:
                missing_doc_count.add(neg_doc)
                tqdm.write(f'missing doc {neg_id}! Skipping') 
                continue
            yield qid, pos_id, query_tok, model.tokenize(pos_doc)
            yield qid, neg_id, query_tok, model.tokenize(neg_doc)
    print("total unique train documents missing in train-qrels", len(missing_doc_count) ) 

def iter_valid_records(model, dataset, run, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_valid_records(model, dataset, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch)


def _iter_valid_records(model, dataset, run):
    ds_queries, ds_docs = dataset
    missing_doc_count = set()
    for qid in run:
        query_tok = model.tokenize(ds_queries[qid])
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                missing_doc_count.add(did)
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc)
            yield qid, did, query_tok, doc_tok
    print("number of missing documents from qrels", len(missing_doc_count))


def _pack_n_ship(batch):
    QLEN = 20
    MAX_DLEN = 800
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
    }


def _pad_crop(items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result).long().cuda()


def _mask(items, l):
    result = []
    for item in items:
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result).float().cuda()

