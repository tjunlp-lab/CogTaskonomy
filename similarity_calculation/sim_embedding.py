from scipy.cluster.hierarchy import dendrogram, ward
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.dsewise import cosine_similarity
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    jaccard_score,
)
from scipy.stats import pearsonr
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from tqdm import tqdm, trange
import pandas as pd
import os
from statsmodels.stats._knockoff import RegressionFDR
import re
import json
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_dir",
        default=None,
        type=str,
        required=True,
        help="The embedding paths obtained by different task-specific models are used to calculate the task similarity of CRA and DSE.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output path of the result.",
    )
    parser.add_argument(
        "--oralresult_dir",
        default="",
        type=str,
        required=False,
        help="Transfer learning result path for evaluating generated similarity.",
    )
    args = parser.parse_args()
    logger.info("The args: {}".format(args))

    modelname_list = [
        "CoLA",
        "QNLI",
        "RTE",
        "MNLI",
        "SST-2",
        "MRPC",
        "STS-B",
        "QQP",
        "QA",
        "NER",
        "IR",
        "RC",
    ]
    modelname_list2 = [
        "CoLA",
        "QNLI",
        "RTE",
        "MNLI",
        "SST-2",
        "MRPC",
        "STS-B",
        "QQP",
        "qa",
        "ner",
        "ms",
        "redn",
    ]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the transfer learning result
    if args.oralresult_dir != "":
        transfer_ranks = {}
        transfer_all = pd.read_csv(args.oralresult_dir, index_col=0,)
        for index, row in transfer_all.iterrows():
            transfer_all.at[index, index] = -1
        transfer_ranks = transfer_all.rank(ascending=False)

    # CRA
    # Use different methods of calculation
    rankofembedding = {}
    bestmetric = ""
    bestrank = 100
    for form in Matrix_form:
        rankofembedding[form] = {}
        for firstep in firsteps:
            rankofembedding[form][firstep] = {}
            RDM = Cal_RDM(args.embedding_dir, form, firstep)
            for secstep in secsteps:
                sim_embedding_cra = cal_embedding_sim(RDM, modelname_list, secstep)
                sim_embedding_cra.to_csv(
                    os.path.join(
                        args.output_dir,
                        "sim_CRA_{0}_{1}_{2}.csv".format(form, firstep, secstep),
                    )
                )
                if args.oralresult_dir != "":
                    rank_cra = calrankvalue(sim_embedding_cra, transfer_ranks)
                    rankofembedding[form][firstep][secstep] = str(rank_cra)
                    if bestrank >= rank_cra[0][0]:
                        bestrank = rank_cra[0][0]
                        bestmetric = "-".join([form, firstep, secstep])
    rankofembedding["best"] = "{0}:{1}".format(bestmetric, bestrank)

    if args.oralresult_dir != "":
        with open(os.path.join(args.output_dir, "CRA.json"), "w") as f:
            json.dump(rankofembedding, f)

    # DSE
    embedding_task = {}
    # Each sentence vector is extracted as a task feature vector and a similarity degree is calculated
    sentence_task = {}
    embedding_path = args.embedding_dir
    # The final similarity matrix
    sim_embedding_dse = None
    dis_embedding_dse = None
    for folder in os.listdir(embedding_path):
        if folder == "original" or folder == "base_627sentences.txt":
            continue
        modelname = re.split("[._]", folder)[0]
        path = os.path.join(embedding_path, folder)
        data = pd.read_csv(path, delimiter=" ")
        embedding_task[modelname] = data.drop(columns=["word"]).values

    for i in range(627):
        for modelname in modelname_list2:
            sentence_task[modelname] = embedding_task[modelname][i, :].reshape(1, -1)
        if sim_embedding_dse is None:
            sim_embedding_dse = calsim(sentence_task, modelname_list)
            dis_embedding_dse = caldis(sentence_task, modelname_list2)
        else:

            sim_embedding_dse += calsim(sentence_task, modelname_list)
            dis_embedding_dse += caldis(sentence_task, modelname_list2)

    sim_embedding_dse /= 627
    dis_embedding_dse /= 627
    sim_embedding_dse.to_csv(args.output_dir, "sim_dse.csv")
    if args.oralresult_dir != "":
        rank_dse = calrankvalue(sim_embedding_dse, transfer_ranks)
        print(rank_dse)
