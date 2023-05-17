# SO_with_CoVer
Code for 2023 ACL Findings Paper *Sentence Ordering with a Coherence Verifier*

## Requirements
+ python==3.8.10
+ torch==1.13.0
+ transformers==4.24.0
+ dgl==0.9.1

## Baselines
+ BERSON: https://github.com/hwxcby/BERSON
+ B-TSort: https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering/

## Data
For the AAN and NIPS data, please contact the authors of [Sentence Ordering and Coherence Modeling using Recurrent Neural Networks](https://arxiv.org/pdf/1611.02654.pdf).

The SIND dataset can be downloaded from the [Visual Storytelling](https://visionandlanguage.net/VIST/dataset.html) website.

And the ROCStory dataset can be download from [here](https://cs.rochester.edu/nlp/rocstories/).

## Code
### CoVer
+ Construct the instance with gradual permutation.
Replace the specific name such as 'roc' with 'dataset'

`python get_pairwise_hier_dataset.py dataset`
+ Train the model.

`sh run_score.sh`

### B-TSort and BERSON with CoVer
If you already have the baseline B-TSort and BERSON model and a CoVer model, you can run `run_rerank.sh` in Topo_CoVer or `run_{dataset}_coherence.sh` in BERSON_CoVer.

Besides, we saved the pairwise scores generated by B-TSort in `Topo_CoVer/pairwise_score`, so that you can directly do the reranking process without having a B-TSort model.

## Model
We provide the pretrained coherence mdoel $Cover$ and reproduced BERSON models for four datasets.

You can download from here: https://drive.google.com/drive/folders/1gHqH3inelArIDPhUIu8XjOAbXcaZSKok?usp=sharing