import os
import argparse
import subprocess
import random
import numpy as np
import tempfile
from tqdm import tqdm
import torch
import torch.nn.functional as F

from rankera import data
import pytrec_eval
from statistics import mean
from collections import defaultdict
from transformers.optimization import AdamW

from rankera import modeling
from transformers import AutoTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count() 
device = torch.device("cuda")

LR =  3e-5
BERT_LR = 3e-5
T2_BERT_LR = 3e-5
MAX_EPOCH = 30
BATCH_SIZE = 32  
BATCH_SIZE_2 = 32
BATCHES_PER_EPOCH = 1 
GRAD_ACC_SIZE = 4
PATIENCE = 10 


mixing_ratio1 = 1
mixing_ratio2 = 1

TASK1_START_STOP= {'start':0 , 'stop':your/stop/point}
TASK2_START_STOP= {'start':0 , 'stop':your/stop/point}


name_of_pretrained_model =  'nlpaueb/legal-bert-base-uncased' 
VALIDATION_METRIC = ('P_5',) 


model = modeling.BertForPairwiseLearning.from_pretrained(name_of_pretrained_model, cache_dir='./cache_dir/')
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(name_of_pretrained_model, cache_dir='./cache_dir/', device=device)

MODEL_MAP = {'bert': model}

#class MultipleOptimizer(object):
    #def __init__(self, *op):
        #self.optimizers = op

    #def zero_grad(self):
        #for op in self.optimizers:
            #op.zero_grad()

    #def step(self):
        #for op in self.optimizers:
            #op.step()
            
def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir=None):

    model = model.cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()
   
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = [v for k, v in params if not k.startswith('bert.')]
    bert_params =[v for k, v in params if k.startswith('bert.')]
    optimizer1 = AdamW(model.parameters(), lr=BERT_LR)
    optimizer2 = AdamW(bert_params, lr=T2_BERT_LR)    


    epoch = 0
    top_valid_score = None
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model,optimizer1, optimizer2, dataset, train_pairs, qrels_train, epoch)
        print(f'train epoch={epoch} loss={loss}')

        valid_score = validate(model, dataset, valid_run, qrels_valid, epoch)
        print(f'validation epoch={epoch} score={valid_score}')

        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            torch.save(model.state_dict(), os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break
        
    if top_valid_score_epoch != epoch:
        model.load_state_dict(torch.load(os.path.join(model_out_dir, 'weights.p')))
    return (model, top_valid_score_epoch)


def train_iteration(model, optimizer1, optimizer2, dataset, train_pairs, qrels, epochnum):
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH + BATCH_SIZE, ncols=80, desc='train', leave=False) as pbar: # change 2460 for total venvadded
        for record in data.iter_train_pairs_longformer(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE, tokenizer):
            count = 0
            if (TASK1_START_STOP['start']<=epochnum) and (epochnum<=TASK1_START_STOP['stop']):
                outputs = model(**record['model_input'].to(device), side_input=record['side_input'].to(device), task=1)

                scores = outputs['logits']
                count = len(record['query_id']) // 2 
                scores = scores.reshape(count, 2)



                loss = torch.mean(-torch.log(scores.softmax(dim=1)[:, 0]))

                loss *= mixing_ratio1
                loss.backward()
                total_loss += loss.item()
                total += count

                if total % BATCH_SIZE == 0:
                    optimizer1.step()
                    optimizer1.zero_grad()
                    
            if (TASK2_START_STOP['start']<=epochnum) and (epochnum<=TASK2_START_STOP['stop']):
                outputs = model(side_input=record['side_input'].to(device), task=2)
                embedds = outputs['embedds']
                distance_metric = 'l2-norm' 
                loss_margin = 1
                d_length=  embedds.shape[1]
                query_vectors = embedds.reshape(-1,2, d_length)[: ,0, :]
                doc_vectors = embedds.reshape(-1,2, d_length)[: ,1, :]

                if distance_metric == 'l2-norm':
                    qv_sim = F.pairwise_distance(query_vectors, doc_vectors)
                    qv_sim = qv_sim.reshape(-1,2)
                    distance_positive = qv_sim[:, 0] #F.pairwise_distance(query, positive)
                    distance_negative = qv_sim[:, 1] # F.pairwise_distance(query, negative)
                    loss2 = torch.mean(F.relu(distance_positive - distance_negative + loss_margin))

                #elif distance_metric == 'cosine':  # independent of length
                    #qv_cosine_sim = F.cosine_similarity(query_vectors, doc_vectors)
                    #qv_cosine_sim = qv_cosine_sim.reshape(-1,2)
                    #distance_positive = qv_cosine_sim[:, 0] # F.cosine_similarity(query, positive)
                    #distance_negative = qv_cosine_sim[:, 1] # F.cosine_similarity(query, negative)
                    #loss2 = torch.mean(F.relu(-distance_positive + distance_negative + loss_margin))

                loss2 *= mixing_ratio2
                loss2.backward()
                if total % BATCH_SIZE_2 == 0:
                    optimizer2.step()
                    optimizer2.zero_grad()  
            pbar.update(count)
#             if total >= (BATCH_SIZE * BATCHES_PER_EPOCH):
#                 return total_loss # kingadded instead of last line in this function
        return total_loss


def validate(model, dataset, run, valid_qrels, epoch):
    run_scores = run_model(model, dataset, run)
    metric = VALIDATION_METRIC
#     if metric.startswith("P_"):
#         metric = "P"
#     trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
#     eval_scores = trec_eval.evaluate(run_scores)
#     return mean([d[VALIDATION_METRIC] for d in eval_scores.values()])
    evaluator = pytrec_eval.RelevanceEvaluator(valid_qrels, set(metric))
    results = evaluator.evaluate(run_scores)

    metric_values = {}
    for measure in sorted(metric):
        res = pytrec_eval.compute_aggregated_measure(
                measure, 
                [query_measures[measure]  for query_measures in results.values()]
            )
        metric_values[measure] = np.round(100 * res, 4)
    return metric_values[VALIDATION_METRIC[0]] # we use the first metric in validation_metric tuple

def run_model(model, dataset, run, desc='valid'):
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records_longformer(model, dataset, run, batch_size=BATCH_SIZE, tokenizer=tokenizer):
            outputs = model(**records['model_input'].to(device))
#             logits = outputs.logits
#             scores = logits[:, 1] # [item[1] for item in logits]
            scores = outputs['logits']
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run
    

def write_run(rerank_run, runf):
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def update_batch_num_info(train_pairs, qrels):
    global BATCHES_PER_EPOCH
    global BATCH_SIZE
    tcounter = 0
    for row in train_pairs:
        for col in train_pairs[row]:
            if qrels.get(row, {}).get(col, 0) > 0:
                tcounter += 1
    
    BATCHES_PER_EPOCH = tcounter//(BATCH_SIZE)

def main_cli():
    parser = argparse.ArgumentParser('Model Fine-tuning and Validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='bert')

    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    
    parser.add_argument('--valid_qrel', type=argparse.FileType('rt'))
    
    
    args = parser.parse_args()
    
    vqrel = data.read_qrels_dict(args.valid_qrel)
    
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)

    update_batch_num_info(train_pairs, qrels) 
    main(model, dataset, train_pairs, qrels, valid_run, vqrel, args.model_out_dir)

if __name__ == '__main__':
    
    main_cli()

