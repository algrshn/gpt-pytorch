# PyTorch implementation of GPT/GPT-2 from scratch from the original papers"Improving Language Understanding by Generative Pre-Training" and "Language Models are Unsupervised Multitask Learners".

Originally published on [*my site*](https://alexgrishin.ai/pytorch_implementation_of_gpt).
<br /><br />

This is a PyTorch implementation of GPT/GPT-2 from the original papers
[*GPT*](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
  and [*GPT-2*](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   (Alec Radford et al.).
  GPT is coded from scratch in "vanilla" PyTorch without use of PyTorch transformer classes.
  The model was trained on a (non negligible) fraction of The Pile dataset.
  The main goal of the project was to see how far one can go in terms of training the GPT family of models
  on a relatively serious size corpus with
  only one 8Gb GPU.

  Only the smallest model can fit in such low GPU memory,
  and 8Gb memory also imposes truly harsh limitations on batch size. I budgeted two months of training time for the project
which forced me to downsize to a 21.5 bln tokens subset of The Pile, on which I could train the model for one epoch.

The resulting undertrained model, however, produces sensible results. If we don't push it far and restrain prompt completions to (a very modest number of) 20 tokens, it generates reasonable completions. After that limit it starts repeating itself - the behaviour also observed in the original GPT. The model achieves perplexity of 19.35 on the validation set.

### Dataset and preprocessing

[*The Pile dataset*](https://arxiv.org/pdf/2101.00027.pdf) is a diverse collection
of 22 subsets of different origins with overall effective size of ~1.3Tb. The whole dataset is split between
thirty jsonl files (0.jsonl, 1.jsonl, ..., 29.jsonl) each of which is about 45Gb in size. Each line in each jsonl file
corresponds to a document from one of 22 subsets. Documents from all subsets are distributed randomly across all jsonl
files, so each file has roughly the same composition of subsets. Each line is a dictionary containing (apart from
other info, which I'm not using) document text and subset name this document belongs to.

My preprocessing procedure consists of 4 steps. Scripts corresponding to these steps use the \[preprocessing\]
section of the config file, which looks like that:
