# CogTaskonomy

### A cognitively inspired framework to learn taxonomy for  NLP  tasks

The  framework  consists  of Cognitive Representation Analytics (CRA) and Cognitive-Neural Mapping (CNM).

Thanks for [CogniVal](https://github.com/DS3Lab/cognival), part of code are revised based on this work.
You can get Cognitive Datasets from [here](https://osf.io/crwz7/)

## CNM(Cognitive-Neural Mapping)

### Embedding maps to cognitive data

In order for the script to run properly the necessary information has to be previously stored inside the config/setupConfig.json.json file. This information consists of the names of the datafiles, the path to where they are stored, the number of hidden layers and nodes for the neural network etc.
An example of the ``setupConfig.json`` with the necessary information to run this case is stored in ``config/example.json``.

For the input format of cognitive data and embedding, see [CogniVal](https://github.com/DS3Lab/cognival).

Command Example:

```
python embedding2cog/script.py config/example.json -c $cogdata_name -f ALL_DIM -w $wordembedding_name -o $word2cog_dir
```

### Get the similarity and evaluate it on transfer learning

The input transfer learning result is required to be a CSV file, which saves a matrix composed of transfer learning results between two tasks. The source task along axis=0 and the target task along axis=1 are the input transfer learning results. A possible example is as follows (bold for target task, unbold for source task):
|       | CoLA   | QNLI   | RTE    | MNLI   | SST-2  | MRPC   | STS-B  | QQP    | NER    | RC     | QA     | IR     |
| ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| CoLA  | 54.96  | 90.81  | 54.87  | 84.05  | 92.43  | 76.47  | 40.95  | 90.17  | 90.58  | 89.62  | 75.92  | 67.79  |
| QNLI  | 43.30  | 90.98  | 66.06  | 83.74  | 92.09  | 81.62  | 66.62  | 90.01  | 90.34  | 89.09  | 76.37  | 61.05  |
| RTE   | 51.81  | 90.57  | 61.01  | 83.92  | 91.97  | 77.94  | 45.50  | 90.19  | 90.28  | 90.16  | 76.18  | 67.01  |
| MNLI  | 48.15  | 90.28  | 74.37  | 83.94  | 93.12  | 82.84  | 49.22  | 89.96  | 90.33  | 89.17  | 76.90  | 67.01  |
| SST-2 | 49.12  | 90.81  | 53.07  | 84.06  | 92.55  | 73.04  | 54.36  | 90.20  | 90.06  | 90.05  | 75.50  | 67.04  |
| MRPC  | 50.46  | 90.54  | 62.82  | 83.89  | 92.20  | 75.25  | 24.40  | 89.73  | 90.22  | 90.53  | 75.79  | 66.41  |
| STS-B | 47.20  | 90.02  | 64.26  | 83.68  | 92.32  | 79.41  | 49.84  | 89.41  | 90.29  | 90.95  | 72.74  | 5.96   |
| QQP   | 50.51  | 90.99  | 62.82  | 83.92  | 92.09  | 80.39  | 42.43  | 91.04  | 90.34  | 90.08  | 75.85  | 66.22  |
| IR    | 43.27  | 91.29  | 62.45  | 83.80  | 91.40  | 79.41  | 53.02  | 90.20  | 90.36  | 89.77  | 77.39  | 65.69  |
| NER   | 48.29  | 88.65  | 61.37  | 83.72  | 92.20  | 73.04  | 41.26  | 89.97  | 90.75  | 89.57  | 75.16  | 62.30  |
| QA    | 43.87  | 90.98  | 67.15  | 83.20  | 90.60  | 76.23  | 50.05  | 89.78  | 90.33  | 89.14  | 74.90  | 64.66  |
| RC    | 40.65  | 90.23  | 56.68  | 83.91  | 91.97  | 70.83  | 30.73  | 89.92  | 90.08  | 89.64  | 76.19  | 67.23  |

Command Example:

```
python similarity_calculation/sim_voxels.py.py -Cognitivedata_dir $cogdata_name --embedding2cog_dir $word2cog_dir --oralresult_dir $transfer_dir --output_dir $output_dir
```

## CRA(Cognitive Representation Analytic)

Cognitive data and embedding maintain the same input format.

Command Example:

```
python similarity_calculation/sim_embedding.py.py -Cognitivedata_dir $cogdata_name  --oralresult_dir $transfer_dir --output_dir $output_dir 
```

# Reference

If you use the source codes here in your work, please cite the corresponding paper. The bibtex is listed below:

```
@inproceedings{luo-etal-2022-cogtaskonomy,
    title = "{C}og{T}askonomy: Cognitively Inspired Task Taxonomy Is Beneficial to Transfer Learning in {NLP}",
    author = "Luo, Yifei  and
      Xu, Minghui  and
      Xiong, Deyi",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.64",
    doi = "10.18653/v1/2022.acl-long.64",
    pages = "904--920",
    abstract = "Is there a principle to guide transfer learning across tasks in natural language processing (NLP)? Taxonomy (Zamir et al., 2018) finds that a structure exists among visual tasks, as a principle underlying transfer learning for them. In this paper, we propose a cognitively inspired framework, CogTaskonomy, to learn taxonomy for NLP tasks. The framework consists of Cognitive Representation Analytics (CRA) and Cognitive-Neural Mapping (CNM). The former employs Representational Similarity Analysis, which is commonly used in computational neuroscience to find a correlation between brain-activity measurement and computational modeling, to estimate task similarity with task-specific sentence representations. The latter learns to detect task relations by projecting neural representations from NLP models to cognitive signals (i.e., fMRI voxels). Experiments on 12 NLP tasks, where BERT/TinyBERT are used as the underlying models for transfer learning, demonstrate that the proposed CogTaxonomy is able to guide transfer learning, achieving performance competitive to the Analytic Hierarchy Process (Saaty, 1987) used in visual Taskonomy (Zamir et al., 2018) but without requiring exhaustive pairwise $O(m^2)$ task transferring. Analyses further discover that CNM is capable of learning model-agnostic task taxonomy.",
}
```
