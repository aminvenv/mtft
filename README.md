# mtft
To train the model:

```
python train.py   --model bert  --datafiles path/to/queries path/to/docs   --qrels path/to/train_qrels  --train_pairs path/to/train_pairs   --valid_run path/to/valid_pairs   --model_out_dir path/for/saving/the/model --valid_qrel path/to/valid_qrels
```

For re-ranking:

```
python rerank.py  --model bert --datafiles path/to/queries path/to/docs  --run path/to/initial_ranking/run --model_weights path/to/the/saved/model --out_path path/for/the/output/run --test_qrels path/to/test_qrels
```


This implementation is based on [CEDR](https://github.com/Georgetown-IR-Lab/cedr)
