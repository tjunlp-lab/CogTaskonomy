from utils import *
import json
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Cognitivedata_dir",
        default=None,
        type=str,
        required=True,
        help="The FMRI data.",
    )
    parser.add_argument(
        "--embedding2cog_dir",
        default=None,
        type=str,
        required=True,
        help="Path file path of cognitive data obtained by embedding prediction.",
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
        required=True,
        help="Transfer learning result path for evaluating generated similarity.",
    )
    args = parser.parse_args()
    logger.info("The args: {}".format(args))
        

    summarized_data = {}
    voxlist = [x for x in range(2500, 32500, 2500)]
    voxlist = [500, 1000, 1500, 2000] + voxlist
    for voxel_nums in voxlist:
        true_dir = os.path.join(args.Cognitivedata_dir, str(voxel_nums))
        rank_ndcg = np.zeros((8, 1))
        (
            sum_df_sim_averagersqs,
            sum_df_sim_averagecorrs,
            sum_averagesim_rsqs,
            sum_averagesim_corrs,
            sum_df_sim_averageeudis,
            sum_df_sim_averagecosine,
            sum_averagesim_eudis,
            sum_averagesim_cosine,
        ) = (None, None, None, None, None, None, None, None)
        summarized_data[voxel_nums] = {"avgsim_rank": {}, "avgrank": {}}
        for seed in [5, 10, 20, 30, 123]:
            predict_dir = (
                args.embedding2cog_dir
                + str(voxel_nums)
                + "-seed"
                + str(seed)
            )
            print(predict_dir)
            outpath = (
                args.output_dir
                + str(voxel_nums)
                + "/"
                + str(seed)
            )
            (
                df_sim_averagersqs,
                df_sim_averagecorrs,
                df_sim_averagecosine,
                df_sim_averageeudis,
                averagesim_rsqs,
                averagesim_corrs,
                averagesim_cosine,
                averagesim_eudis,
            ) = task_sim(
                outpath,
                predict_dir,
                true_dir,
                voxel_nums,
            )
            if sum_df_sim_averagecorrs is None:
                sum_df_sim_averagecorrs = df_sim_averagecorrs
                sum_df_sim_averagersqs = df_sim_averagersqs
                sum_averagesim_corrs = averagesim_corrs
                sum_averagesim_rsqs = averagesim_rsqs
                sum_df_sim_averageeudis = df_sim_averageeudis
                sum_df_sim_averagecosine = df_sim_averagecosine
                sum_averagesim_eudis = averagesim_eudis
                sum_averagesim_cosine = averagesim_cosine
            else:
                sum_df_sim_averagecorrs += df_sim_averagecorrs
                sum_df_sim_averagersqs += df_sim_averagersqs
                sum_averagesim_corrs += averagesim_corrs
                sum_averagesim_rsqs += averagesim_rsqs
                sum_df_sim_averageeudis += df_sim_averageeudis
                sum_df_sim_averagecosine += df_sim_averagecosine
                sum_averagesim_eudis += averagesim_eudis
                sum_averagesim_cosine += averagesim_cosine

            transfer_all = pd.read_csv(
                args.oralresult_dir,
                index_col=0,
            )

            for index, row in transfer_all.iterrows():
                transfer_all.at[index, index] = -1
            transfer_rank_ndcg = transfer_all.rank(ascending=False)


            # ndcg

            temp_ndcg = topNinM(
                [
                    df_sim_averagersqs,
                    df_sim_averagecorrs,
                    averagesim_rsqs,
                    averagesim_corrs,
                    df_sim_averagecosine,
                    df_sim_averageeudis,
                    averagesim_cosine,
                    averagesim_eudis,
                ],
                transfer_rank_ndcg,
            )
            rank_ndcg += temp_ndcg
        rank_ndcg /= 5

        summarized_data[voxel_nums]["avgrank"]["rank_ndcg"] = {
            "sim_averagersqs": rank_ndcg[0, :].tolist(),
            "sim_averagecorrs": rank_ndcg[1, :].tolist(),
            "averagesim_rsqs": rank_ndcg[2, :].tolist(),
            "averagesim_corrs": rank_ndcg[3, :].tolist(),
            "sim_averagecosine": rank_ndcg[4, :].tolist(),
            "sim_averageeudis": rank_ndcg[5, :].tolist(),
            "averagesim_cosine": rank_ndcg[6, :].tolist(),
            "averagesim_eudis": rank_ndcg[7, :].tolist(),
        }

        summarized_data[voxel_nums]["task"] = []

        print("The target task similarity task was obtained by averaging the task similarity degree of each round ")
        print(summarized_data[voxel_nums]["task"])
        rank_ndcg = topNinM(
            [
                sum_df_sim_averagersqs / 5,
                sum_df_sim_averagecorrs / 5,
                sum_averagesim_rsqs / 5,
                sum_averagesim_corrs / 5,
                sum_df_sim_averagecosine / 5,
                sum_df_sim_averageeudis / 5,
                sum_averagesim_cosine / 5,
                sum_averagesim_eudis / 5,
            ],
            transfer_rank_ndcg,
        )

        summarized_data[voxel_nums]["avgsim_rank"]["rank_ndcg"] = {
            "sim_averagersqs": rank_ndcg[0, :].tolist(),
            "sim_averagecorrs": rank_ndcg[1, :].tolist(),
            "averagesim_rsqs": rank_ndcg[2, :].tolist(),
            "averagesim_corrs": rank_ndcg[3, :].tolist(),
            "sim_averagecosine": rank_ndcg[4, :].tolist(),
            "sim_averageeudis": rank_ndcg[5, :].tolist(),
            "averagesim_cosine": rank_ndcg[6, :].tolist(),
            "averagesim_eudis": rank_ndcg[7, :].tolist(),
        }
    with open(
        args.output_dir,
        "w",
    ) as f:
        json.dump(summarized_data, f)
