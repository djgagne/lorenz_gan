import argparse
import yaml
from multiprocessing import Pool
import pandas as pd
import numpy as np
import traceback
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="lorenz.yaml", help="Config yaml file")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    gan_indices = config["gan_indices"]
    data_file = config["data_file"]
    jobs = []
    pool = Pool(args.nprocs)
    for gan_index in gan_indices:
        jobs.append(pool.apply_async(run_offline_analysis, (gan_index, data_file, config["gan_path"],
                                  config["seed"], config["batch_size"], config["out_path"], config["meta_columns"])))
    pool.close()
    pool.join()
    return


def run_offline_analysis(gan_index, data_file, gan_path, seed, batch_size,
                         out_dir, meta_columns,
                         pdf_bins=np.arange(-16, 23, 0.1),
                         bandwidth=0.2,
                         time_lags=np.arange(1, 500), x_index=0):
    try:
        from lorenz_gan.analysis import offline_gan_predictions, calc_pdf_hist, hellinger, time_correlations
        data = pd.read_csv(data_file, dtype='float32')
        epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30],
                        dtype=np.int32)
        print("Calculate offline GAN predictions {0:03d}".format(gan_index))
        gan_preds, gan_noise = offline_gan_predictions(gan_index, data, gan_path, seed=seed, batch_size=batch_size)
        print("Saving offline GAN predictions {0:03d}".format(gan_index))
        for p_type in gan_preds.keys():
            pd.merge(data[meta_columns], gan_preds[p_type], left_index=True, right_index=True).to_csv(
                join(out_dir, "gan_{0:03d}_{1}_offline_predictions.csv".format(gan_index, p_type)), index_label="Index")
        gan_noise.to_csv(join(out_dir, "gan_{0:03d}_noise_corr.csv".format(gan_index)), index_label="Model")
        pdf_columns = ["truth"]
        for key in gan_preds.keys():
            pdf_columns += [key + "_" + x for x in gan_preds[key].columns]
        pdfs = pd.DataFrame(0.0, index=pdf_bins[:-1], columns=pdf_columns, dtype=np.float32)
        print("Calc PDFs of GAN predictions {0:03d}".format(gan_index))
        pdfs.loc[:, "truth"] = calc_pdf_hist(data["Ux_t+1"].values, pdf_bins)
        for key in gan_preds.keys():
            for col in gan_preds[key].columns:
                pdfs.loc[:, key + "_" + col] = calc_pdf_hist(gan_preds[key][col].values, pdf_bins)
        pdfs.to_csv(join(out_dir, "gan_{0:03d}_offline_pdfs.csv".format(gan_index)), index_label="Bins")
        print("Calc Hellingers of GAN predictions {0:03d}".format(gan_index))
        hellingers = pd.DataFrame(0.0, index=epochs,
                                columns=["{0:04d}_{1}".format(gan_index, k) for k in gan_preds.keys()],
                                dtype=np.float32)
        for key in gan_preds.keys():
            for c, col in enumerate(gan_preds[key].columns):
                hellingers.loc[epochs[c], "{0:04d}_{1}".format(gan_index, key)] = hellinger(pdf_bins[:-1],
                                                                                            pdfs["truth"].values,
                                                                                            pdfs[key + "_" + col].values)
        hellingers.to_csv(join(out_dir, "gan_{0:03d}_offline_hellinger.csv".format(gan_index)), index_label="Epoch")
        print("Calc time correlations of GAN predictions {0:03d}".format(gan_index))
        gan_time_corr = pd.DataFrame(0.0, index=time_lags, columns=pdfs.columns, dtype=np.float32)
        x_points = data["x_index"] == x_index
        for col in gan_time_corr.columns:
            if col == "truth":
                gan_time_corr.loc[:, col] = time_correlations(data.loc[x_points, "Ux_t+1"], time_lags)
            else:
                key = col.split("_")[0]
                mod = col[len(key) + 1:]
                gan_time_corr.loc[:, col] = time_correlations(gan_preds[key].loc[x_points, mod], time_lags)
        gan_time_corr.to_csv(join(out_dir, "gan_{0:03d}_time_correlations.csv".format(gan_index)), 
                             index_label="Time Lag")
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return
if __name__ =="__main__":
    main()
