import argparse
from rankera import train
from rankera import data
import torch
import os
import pandas as pd



def main_cli():
    parser = argparse.ArgumentParser('re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='bertcat')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--test_qrels', type=argparse.FileType('rt'))

    args = parser.parse_args()
    model = train.model

    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load_state_dict(torch.load(args.model_weights.name))
        
    test_qrels = data.read_qrels_dict(args.test_qrels)
    res = train.run_model(model, dataset, run, desc='rerank')

    q_list = []
    d_list = []
    rel_list = []
    for i in res:
        for j in res[i]:
            q_list.append(i)
            d_list.append(j)
            rel_list.append(res[i][j])
    res_df = pd.DataFrame(columns=['qid', 'Q0', 'docid', 'rank', 'score', 'name'])
    res_df['qid'] = q_list
    res_df['Q0'] = len(q_list) * ['Q0']
    res_df['docid'] = d_list
    res_df['rank'] = 0
    res_df['score'] = rel_list
    res_df['name'] =  len(q_list) * ['bert']
    
    res_df.to_csv(args.out_path, index=False, header=False, sep='\t')
    
if __name__ == '__main__':
    main_cli()
