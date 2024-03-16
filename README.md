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

**\[preprocessing\]**<br>
path_to_original_data=/media/alex/data2/GPT2_data/ThePile/pile/train/<br>
path_to_store_preprocessed_data=/media/alex/data1/GPT2_data/data/<br>
pile_sets_to_use=FreeLaw|HackerNews|Books3|OpenWebText2|Gutenberg (PG-19)|Pile-CC|BookCorpus2|Wikipedia (en)|PubMed Abstracts<br>
max_len_words=256<br>
min_len_words=16<br>SS
max_len_tokens=512<br>
max_num_of_tokens_in_batch=2100<br>

Below is a description of preprocessing scripts.

#### Step 0 - filter and truncate

As my computational resourses were
extremely limited, I decided not even to attempt to train the model on anything but plain English. So, I
filtered out subsets which are heavy on code examples, math equations, chemical formulas, etc.
The parameter pile_sets_to_use in the config file controls what subsets will be used.
These are the subsets I kept (in size descending order):
<ul>
<li>Pile-CC</li>
<li>Books3</li>
<li>OpenWebText2</li>
<li>FreeLaw</li>
<li>PubMed Abstracts</li>
<li>Gutenberg (PG-19)</li>
<li>Wikipedia (en)</li>
<li>BookCorpus2</li>
<li>HackerNews</li>
</ul>
They comprise roughly 55% of the full Pile dataset.

Having in mind that I will later impose restrictions on context size in tokens,
I truncated documents with more than max_len_words (256 in my case) words, which was a substantial cut.
I also threw away all documents which had less than min_len_words (16 in my case) words in them (there were
very few of those).

Here's how to run step 0:

<div class="code_box"><code>$ python3 0_filter_sets_truncate_long.py --chunk_num=0</code></div>

The command line parameter --chunk_num can take values from 0 to 29 and corresponds to different
  original jsonl files (chunks).
