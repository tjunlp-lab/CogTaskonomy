import networkx as nx
from scipy.cluster.hierarchy import dendrogram, ward
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, pdist
import pandas as pd
import os

outpath = ""
subjects = ["M02", "M07", "M15", "P01", "M04"]
seed = ["5", "10", "20", "30", "123"]

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


def rename(nameA):
    if nameA == "qa":
        nameA = "QA"
    elif nameA == "redn":
        nameA = "RC"
    elif nameA == "ms":
        nameA = "IR"
    elif nameA == "ner":
        nameA = "NER"
    return nameA


def cal_embedding_sim(mat_corrs, modelname_list, secstep="cosine"):
    df_sim = pd.DataFrame(
        np.zeros(144).reshape(12, 12), columns=[i for i in modelname_list]
    )
    df_sim.index = [i for i in modelname_list]
    for nameA, row1 in mat_corrs.items():
        nameA = rename(nameA)
        for nameB, row2 in mat_corrs.items():
            nameB = rename(nameB)
            #             row2.fillna(0, inplace=True)
            #             row1.fillna(0, inplace=True)
            row1[np.isnan(row1)] = 0
            row2[np.isnan(row2)] = 0
            if secstep == "cosine":
                try:
                    sim = cosine_similarity(row1.reshape(1, -1), row2.reshape(1, -1))
                    if nameA != nameB:
                        df_sim.at[nameA, nameB] = sim[0][0]
                    df_sim.at[nameB, nameA] = sim[0][0]
                except:
                    print(row1)
                    print(row2)
            elif secstep == "pearsonr":
                sim, _ = st.pearsonr(row1, row2)
                if nameA != nameB:
                    df_sim.at[nameA, nameB] = sim
                df_sim.at[nameB, nameA] = sim
            elif secstep == "jaccard":
                sim = jaccard_score(row1, row2)
                if nameA != nameB:
                    df_sim.at[nameA, nameB] = sim
                df_sim.at[nameB, nameA] = sim
            else:
                try:
                    sim, _ = st.spearmanr(row1, row2)
                    if nameA != nameB:
                        df_sim.at[nameA, nameB] = sim
                    df_sim.at[nameB, nameA] = sim
                except:
                    print(row1)
                    print(row2)

    return df_sim


def Cal_RDM(embedding_path, Matrix_form, firstep):
    dic_taskrepr = {}
    for folder in os.listdir(embedding_path):
        if folder == "original" or folder == "base_627sentences.txt":
            continue
        modelname = re.split("[._]", folder)[0]
        path = os.path.join(embedding_path, folder)
        data = pd.read_csv(path, delimiter=" ")
        if Matrix_form == "RDM":
            dic_taskrepr[modelname] = pdist(data.drop(columns=["word"]).values, firstep)
        else:
            dic_taskrepr[modelname] = 1 - pdist(
                data.drop(columns=["word"]).values, firstep
            )

    return dic_taskrepr


def calsim(mat_corrs, modelname_list, secstep):
    df_sim = pd.DataFrame(
        np.zeros(144).reshape(12, 12), columns=[i for i in modelname_list]
    )
    df_sim.index = [i for i in modelname_list]
    for nameA, row1 in mat_corrs.items():
        nameA = rename(nameA)
        for nameB, row2 in mat_corrs.items():
            nameB = rename(nameB)
            #             row2.fillna(0, inplace=True)
            #             row1.fillna(0, inplace=True)
            row1[np.isnan(row1)] = 0
            row2[np.isnan(row2)] = 0
            try:
                sim = cosine_similarity(row1, row2)
            except:
                print(row1)
                print(row2)
            if nameA != nameB:
                df_sim.at[nameA, nameB] = sim[0][0]
            df_sim.at[nameB, nameA] = sim[0][0]
    return df_sim


def caldis(mat_corrs, modelname_list, voxel_nums):
    dis = []
    for nameA in modelname_list:
        dis.append(mat_corrs[nameA])
    dis = np.array(dis).reshape(12, voxel_nums)
    return pdist(dis, "cosine")


def heatmap(df_sim, outputname):
    plt.rcParams["figure.figsize"] = (8.0, 6.4)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    heatmap_file_path = os.path.join(outpath, outputname)
    sns.set()
    sns.heatmap(df_sim, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    plt.title(outputname)
    plt.savefig(heatmap_file_path)
    plt.close()


def tasktree(df_dis, outputname, labels=None):
    linkage_array = ward(df_dis)
    if labels:
        dendrogram(linkage_array, labels=labels)
    else:
        dendrogram(linkage_array, labels=df_dis.index)
    plt.title(outputname)
    plt.savefig(os.path.join(outpath, outputname))
    plt.close()


def scale_width(arr, alpha, beta):
    vmin = np.min(arr)
    vmax = np.max(arr)
    new_arr = (arr - vmin) / (vmax - vmin)
    return new_arr * alpha + beta


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors

    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def tasknetwork(df_sim, outputname):
    G = nx.from_pandas_adjacency(df_sim)
    widths = [G[u][v]["weight"] for u, v in G.edges()]
    widths = scale_width(widths, alpha=10, beta=1)
    G = nx.drawing.nx_agraph.to_agraph(G)
    pos = nx.circular_layout(G)

    fig = plt.figure()
    nx.draw(
        G,
        pos,
        node_color="silver",
        node_size=1000,
        font_size=10,
        font_weight="bold",
        width=widths,
        # edge_color="#A0CBE2",
        edge_color=np.array(widths),
        edge_cmap=truncate_colormap(plt.cm.YlGnBu, 0.3, 0.8),
        with_labels=True,
    )
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    plt.title(outputname)
    fig.savefig(os.path.join(outpath, outputname))
    plt.close()


def task_sim(outpathdir, predict_dir, true_dir, voxel_nums):
    # It is used to store the sum of the similarity space calculated by all subjects and to calculate the average value if needed later
    global outpath
    outpath = outpathdir
    if os.path.exists(os.path.join(outpath, "sim_averagersqs.csv")):
        df_sim_averagersqs = pd.read_csv(
            os.path.join(outpath, "sim_averagersqs.csv"), index_col=0
        )
        df_sim_averagecorrs = pd.read_csv(
            os.path.join(outpath, "sim_averagecorrs.csv"), index_col=0
        )
        df_sim_averagecosine = pd.read_csv(
            os.path.join(outpath, "sim_averagecosine.csv"), index_col=0
        )
        df_sim_averageeudis = pd.read_csv(
            os.path.join(outpath, "sim_averageeudis.csv"), index_col=0
        )
        averagesim_rsqs = pd.read_csv(
            os.path.join(outpath, "averagesim_rsqs.csv"), index_col=0
        )
        averagesim_corrs = pd.read_csv(
            os.path.join(outpath, "averagesim_corrs.csv"), index_col=0
        )
        averagesim_cosine = pd.read_csv(
            os.path.join(outpath, "averagesim_cosine.csv"), index_col=0
        )
        averagesim_eudis = pd.read_csv(
            os.path.join(outpath, "averagesim_eudis.csv"), index_col=0
        )
        return [
            df_sim_averagersqs,
            df_sim_averagecorrs,
            df_sim_averagecosine,
            df_sim_averageeudis,
            averagesim_rsqs,
            averagesim_corrs,
            averagesim_cosine,
            averagesim_eudis,
        ]
    else:
        sumsim_rsqs = None
        sumsim_corrs = None
        sumsim_cosine = None
        sumsim_eudis = None
        # A vector used to store the sum of predicted correlation coefficients for each task in a cross-subject data set
        sumrsqs_task = {}
        sumcorrs_task = {}
        sumcosine_task = {}
        sumeudis_task = {}
        # It is used to store the sum of the distance space calculated by all subjects and to calculate the average value if needed later
        sumdis_rsqs = None
        sumdis_corrs = None
        sumdis_cosine = None
        sumdis_eudis = None
        for dataset in datasets:
            predict_voxels = {}
            true_path = os.path.join(true_dir, dataset + "_627sentences.txt")
            true_voxels = (
                pd.read_csv(true_path, delimiter=" ").drop(columns=["word"]).values
            )
            dic_rsqs = {}
            dic_corrs = {}
            dic_cosine = {}
            dic_eudis = {}
            for folder in os.listdir(predict_dir):
                startdir = "pereira-" + str(voxel_nums) + "-" + dataset
                if folder.startswith(startdir):
                    modelname = folder.split("_")[-2]
                    if modelname == "base" or modelname == "random":
                        continue
                    path = os.path.join(predict_dir, folder)
                    filename = folder + ".txt"
                    path = os.path.join(path, filename)
                    data = pd.read_csv(path, delimiter=" ")
                    predict_voxels[modelname] = data.drop(columns=["word"]).values
                    rsqs = np.array(
                        [
                            r2_score(true_voxels[:, i], predict_voxels[modelname][:, i])
                            for i in range(true_voxels.shape[1])
                        ]
                    )
                    corrs = np.array(
                        [
                            pearsonr(true_voxels[:, i], predict_voxels[modelname][:, i])
                            for i in range(true_voxels.shape[1])
                        ]
                    )
                    cosine = np.array(
                        [
                            cosine_similarity(
                                true_voxels[:, i].reshape(1, -1),
                                predict_voxels[modelname][:, i].reshape(1, -1),
                            )[0][0]
                            for i in range(true_voxels.shape[1])
                        ]
                    )
                    eudis = np.array(
                        [
                            euclidean_distances(
                                true_voxels[:, i].reshape(1, -1),
                                predict_voxels[modelname][:, i].reshape(1, -1),
                            )[0][0]
                            for i in range(true_voxels.shape[1])
                        ]
                    )

                    dic_rsqs[modelname] = rsqs.reshape(1, -1)
                    dic_corrs[modelname] = corrs[:, 0].reshape(1, -1)
                    dic_cosine[modelname] = cosine.reshape(1, -1)
                    dic_eudis[modelname] = eudis.reshape(1, -1)
                    if modelname in sumrsqs_task.keys():
                        sumrsqs_task[modelname] += rsqs.reshape(1, -1)
                        sumcorrs_task[modelname] += corrs[:, 0].reshape(1, -1)
                        sumcosine_task[modelname] += cosine.reshape(1, -1)
                        sumeudis_task[modelname] += eudis.reshape(1, -1)
                    else:
                        sumrsqs_task[modelname] = rsqs.reshape(1, -1)
                        sumcorrs_task[modelname] = corrs[:, 0].reshape(1, -1)
                        sumcosine_task[modelname] = cosine.reshape(1, -1)
                        sumeudis_task[modelname] = eudis.reshape(1, -1)

            df_sim_corrs = calsim(dic_corrs, modelname_list)
            df_sim_rsqs = calsim(dic_rsqs, modelname_list)
            dis_corrs = caldis(dic_corrs, modelname_list2, voxel_nums)
            dis_rsqs = caldis(dic_rsqs, modelname_list2, voxel_nums)
            heatmap(df_sim_rsqs, dataset + "-heatmap-rsqs")
            heatmap(df_sim_corrs, dataset + "-heatmap-corrs")
            df_sim_eudis = calsim(dic_eudis, modelname_list)
            df_sim_cosine = calsim(dic_cosine, modelname_list)
            dis_eudis = caldis(dic_eudis, modelname_list2, voxel_nums)
            dis_cosine = caldis(dic_cosine, modelname_list2, voxel_nums)
            heatmap(df_sim_cosine, dataset + "-heatmap-cosine")
            heatmap(df_sim_eudis, dataset + "-heatmap-eudis")
            if sumsim_rsqs is None:
                sumsim_rsqs = df_sim_rsqs
                sumsim_corrs = df_sim_corrs
                sumdis_corrs = dis_corrs
                sumdis_rsqs = dis_rsqs
                sumsim_cosine = df_sim_cosine
                sumsim_eudis = df_sim_eudis
                sumdis_eudis = dis_eudis
                sumdis_cosine = dis_cosine
            else:
                sumsim_rsqs += df_sim_rsqs
                sumsim_corrs += df_sim_corrs
                sumdis_corrs += dis_corrs
                sumdis_rsqs += dis_rsqs
                sumsim_cosine = df_sim_cosine
                sumsim_eudis = df_sim_eudis
                sumdis_eudis = dis_eudis
                sumdis_cosine = dis_cosine

        averagersqs_task = {}
        averagecorrs_task = {}
        averagecosine_task = {}
        averageeudis_task = {}

        for nameA in sumrsqs_task.keys():
            averagersqs_task[nameA] = sumrsqs_task[nameA] * 0.2
            averagecorrs_task[nameA] = sumcorrs_task[nameA] * 0.2
            averagecosine_task[nameA] = sumcosine_task[nameA] * 0.2
            averageeudis_task[nameA] = sumeudis_task[nameA] * 0.2

        df_sim_averagecorrs = calsim(averagecorrs_task, modelname_list)
        df_sim_averagersqs = calsim(averagersqs_task, modelname_list)

        df_sim_averagersqs.to_csv(os.path.join(outpath, "sim_averagersqs.csv"))
        df_sim_averagecorrs.to_csv(os.path.join(outpath, "sim_averagecorrs.csv"))

        dis_averagersqs = caldis(averagersqs_task, modelname_list2, voxel_nums)
        dis_averagecorrs = caldis(averagecorrs_task, modelname_list2, voxel_nums)

        heatmap(df_sim_averagersqs, "heatmap-averagersqs")
        heatmap(df_sim_averagecorrs, "heatmap-averagecorrs")

        tasktree(dis_averagersqs, "tasktree-averagersqs", modelname_list)
        tasktree(dis_averagecorrs, "tasktree-averagecorrs", modelname_list)

        averagesim_rsqs = sumsim_rsqs * 0.2
        averagesim_corrs = sumsim_corrs * 0.2
        averagedis_corrs = sumdis_corrs * 0.2
        averagedis_rsqs = sumdis_rsqs * 0.2
        heatmap(averagesim_rsqs, "averagesim-heatmap-rsqs")
        heatmap(averagesim_corrs, "averagesim-heatmap-corrs")
        averagesim_rsqs.to_csv(os.path.join(outpath, "averagesim_rsqs.csv"))
        averagesim_corrs.to_csv(os.path.join(outpath, "averagesim_corrs.csv"))

        tasktree(averagedis_rsqs, "averagedis-tasktree-rsqs", modelname_list)
        tasktree(averagedis_corrs, "averagedis-tasktree-corrs", modelname_list)

        df_sim_averageeudis = calsim(averageeudis_task, modelname_list)
        df_sim_averagecosine = calsim(averagecosine_task, modelname_list)
        df_sim_averagecosine.to_csv(os.path.join(outpath, "sim_averagecosine.csv"))
        df_sim_averageeudis.to_csv(os.path.join(outpath, "sim_averageeudis.csv"))

        dis_averagecosine = caldis(averagecosine_task, modelname_list2, voxel_nums)
        dis_averageeudis = caldis(averageeudis_task, modelname_list2, voxel_nums)

        heatmap(df_sim_averagecosine, "heatmap-averagecosine")
        heatmap(df_sim_averageeudis, "heatmap-averageeudis")

        tasktree(dis_averagecosine, "tasktree-averagecosine", modelname_list)
        tasktree(dis_averageeudis, "tasktree-averageeudis", modelname_list)

        averagesim_cosine = sumsim_cosine * 0.2
        averagesim_eudis = sumsim_eudis * 0.2
        averagedis_eudis = sumdis_eudis * 0.2
        averagedis_cosine = sumdis_cosine * 0.2
        heatmap(averagesim_cosine, "averagesim-heatmap-cosine")
        heatmap(averagesim_eudis, "averagesim-heatmap-eudis")
        averagesim_cosine.to_csv(os.path.join(outpath, "averagesim_cosine.csv"))
        averagesim_eudis.to_csv(os.path.join(outpath, "averagesim_eudis.csv"))

        tasktree(averagedis_cosine, "averagedis-tasktree-cosine", modelname_list)
        tasktree(averagedis_eudis, "averagedis-tasktree-eudis", modelname_list)

        return (
            df_sim_averagersqs,
            df_sim_averagecorrs,
            df_sim_averagecosine,
            df_sim_averageeudis,
            averagesim_rsqs,
            averagesim_corrs,
            averagesim_cosine,
            averagesim_eudis,
        )


def subject_sim(outpathdir, predict_dir, true_dir, voxel_nums, subject=None):
    # It is used to store the sum of the similarity space calculated by all subjects and to calculate the average value if needed later
    global outpath
    outpath = outpathdir
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    predict_voxels = {}
    true_path = os.path.join(true_dir, subject + "_627sentences.txt")
    true_voxels = pd.read_csv(true_path, delimiter=" ").drop(columns=["word"]).values
    dic_rsqs = {}
    dic_corrs = {}
    dic_cosine = {}
    dic_eudis = {}
    for folder in os.listdir(predict_dir):
        startdir = "pereira-" + str(voxel_nums) + "-" + subject
        if folder.startswith(startdir):
            modelname = folder.split("_")[-2]
            if modelname == "base" or modelname == "random":
                continue
            path = os.path.join(predict_dir, folder)
            filename = folder + ".txt"
            path = os.path.join(path, filename)
            data = pd.read_csv(path, delimiter=" ")
            predict_voxels[modelname] = data.drop(columns=["word"]).values
            rsqs = np.array(
                [
                    r2_score(true_voxels[:, i], predict_voxels[modelname][:, i])
                    for i in range(true_voxels.shape[1])
                ]
            )
            corrs = np.array(
                [
                    pearsonr(true_voxels[:, i], predict_voxels[modelname][:, i])
                    for i in range(true_voxels.shape[1])
                ]
            )
            cosine = np.array(
                [
                    cosine_similarity(
                        true_voxels[:, i].reshape(1, -1),
                        predict_voxels[modelname][:, i].reshape(1, -1),
                    )[0][0]
                    for i in range(true_voxels.shape[1])
                ]
            )
            eudis = np.array(
                [
                    euclidean_distances(
                        true_voxels[:, i].reshape(1, -1),
                        predict_voxels[modelname][:, i].reshape(1, -1),
                    )[0][0]
                    for i in range(true_voxels.shape[1])
                ]
            )

            dic_rsqs[modelname] = rsqs.reshape(1, -1)
            dic_corrs[modelname] = corrs[:, 0].reshape(1, -1)
            dic_cosine[modelname] = cosine.reshape(1, -1)
            dic_eudis[modelname] = eudis.reshape(1, -1)

    df_sim_corrs = calsim(dic_corrs, modelname_list)
    df_sim_rsqs = calsim(dic_rsqs, modelname_list)
    df_sim_rsqs.to_csv(os.path.join(outpath, "sim_rsqs.csv"))
    df_sim_corrs.to_csv(os.path.join(outpath, "sim_corrs.csv"))

    df_sim_eudis = calsim(dic_eudis, modelname_list)
    df_sim_cosine = calsim(dic_cosine, modelname_list)
    df_sim_eudis.to_csv(os.path.join(outpath, "sim_eudis.csv"))
    df_sim_cosine.to_csv(os.path.join(outpath, "sim_cosine.csv"))


def calrankvalue(df_sims, transfer_rank, details=False, summarized_data=None):
    sim_task = {}
    results = []
    for df_sim in df_sims:
        for index, row in df_sim.iterrows():
            maxsim = 0
            for i, v in row.items():
                if v > maxsim and index != i:
                    maxsim = v
                    sim_task[index] = i

        sum1 = 0
        sum2 = 0
        sum3 = 0
        if details:
            summarized_data.append(sim_task)
        for i, v in sim_task.items():

            sum1 += (transfer_rank.at[i, v] + transfer_rank.at[v, i]) / 2
            sum2 += transfer_rank.at[v, i]
            sum3 += transfer_rank.at[i, v]

        sum1 /= 12
        sum2 /= 12
        sum3 /= 12

        results.append([sum2, sum3, sum1])

    return np.array(results)


def topNinM(df_sims, transfer_rank, N=3, M=6):
    sim_task = {}
    results = []
    for df_sim in df_sims:
        for index, row in df_sim.iterrows():
            sim_task[index] = []
            for top in range(N):
                maxsim = 0
                current_task = ""
                for i, v in row.items():
                    if v > maxsim and index != i and (i not in sim_task[index]):
                        maxsim = v
                        current_task = i
                    print(row)
                sim_task[index].append(current_task)
        probability = 0
        print(sim_task)
        for i, toptasks in sim_task.items():
            hit = 0
            for task in toptasks:
                if transfer_rank.at[task, i] <= M:
                    hit += 1
            probability += hit / N

        probability /= 12

        results.append([probability])

    return np.array(results)
