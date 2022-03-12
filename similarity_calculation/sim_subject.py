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
        "--voxel_nums",
        default=25000,
        type=int,
        required=True,
        help="At what voxel level to calculate.",
    )
    args = parser.parse_args()
    logger.info("The args: {}".format(args))
    voxel_nums = args.voxel_nums
    true_dir = os.path.join(args.Cognitivedata_dir, str(voxel_nums))

    for subject in ["M02", "M07", "M15", "P01", "M04"]:
        for seed in [5, 10, 20, 30, 123]:
            outpath = (
                args.output_dir + str(voxel_nums) + "/" + subject + "/" + str(seed)
            )
            predict_dir = args.embedding2cog_dir + str(voxel_nums) + "-seed" + str(seed)
            print(outpath)
            subject_sim(outpath, predict_dir, true_dir, voxel_nums, subject=subject)
