import pandas as pd
import numpy as np
from lorenz_gan.submodels import SubModelGAN, AR1RandomUpdater
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from glob import glob
from os.path import join
import keras.backend as K


def load_test_data(filename,
                   input_columns=("X_t", "Ux_t"),
                   output_columns=("Ux_t+1",),
                   meta_columns=("x_index", "time", "step")):
    all_columns = np.concatenate([list(meta_columns), list(input_columns), list(output_columns)], axis=0)
    data = pd.read_csv(filename, usecols=all_columns)
    return data


def hellinger(a, b):
    """
    Calculate hellinger distance on 2 discrete PDFs a and b.

    Args:
        a:
        b:

    Returns:

    """
    return np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / np.sqrt(2)


def offline_gan_predictions(gan_index, data,
                            gan_path, seed=12421, batch_size=1024):
    rs = np.random.RandomState(seed)
    gen_files = sorted(glob(join(gan_path, "gan_generator_{0:04d}_*.h5".format(gan_index))))
    gen_filenames = [gf.split("/")[-1] for gf in gen_files]
    if gan_index < 300:
        rand_size = 1
    else:
        rand_size = 17
    sess = K.tf.Session(config=K.tf.ConfigProto(intra_op_paralellism_threads=1,
                                                inter_op_paralellism_threads=2,
                                                gpu_options=K.tf.GPUOptions(allow_growth=True)))
    K.set_session(sess)
    K.tf.set_random_seed(seed)
    random_values = rs.normal(size=(data.shape[0], rand_size))
    all_zeros = np.zeros((data.shape[0], rand_size))
    gen_preds = dict()
    for pred_type in ["det", "rand", "corr"]:
        gen_preds[pred_type] = pd.DataFrame(0, index=data.index, columns=gen_filenames,
                             dtype=np.float32)
    gen_noise = pd.DataFrame(0.0, dtype=np.float32, index=gen_filenames, columns=["corr", "noise_sd"])
    with K.tf.device("/cpu:0"):
        for g, gen_file in enumerate(gen_files):
            print("Predicting " + gen_filenames[g])
            gen_model = SubModelGAN(gen_file)
            if gen_model.x_scaling_values.shape[0] == 1:
                gen_preds["det"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t"]],
                                                                                    all_zeros, batch_size=batch_size,
                                                                                    stochastic=0)
                gen_preds["rand"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t"]],
                                                                                     random_values,
                                                                                     batch_size=batch_size,
                                                                                     stochastic=1)
                ar1 = AR1RandomUpdater()
                ar1.fit(gen_preds["det"].loc[data["x_index"] == 0])
                gen_noise.loc[gen_filenames[g]] = [ar1.corr, ar1.noise_sd]
                corr_noise = np.zeros((data.shape[0], rand_size), dtype=np.float32)
                corr_noise[0] = rs.normal(size=(1, rand_size))
                for i in range(1, corr_noise.shape[0]):
                    corr_noise[i] = ar1.update(corr_noise[i - 1], rs)
                gen_preds["corr"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t"]],
                                                                                     corr_noise, batch_size=batch_size,
                                                                                     stochastic=1)
            else:
                gen_preds["det"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t", "Ux_t"]],
                                                                                    all_zeros, batch_size=batch_size,
                                                                                    stochastic=0)
                gen_preds["rand"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t", "Ux_t"]],
                                                                                     random_values,
                                                                                     batch_size=batch_size,
                                                                                     stochastic=1)
                ar1 = AR1RandomUpdater()
                x_indices = data["x_index"] == 0
                ar1.fit(data.loc[x_indices, "Ux_t+1"] - gen_preds["det"].loc[x_indices, gen_filenames[g]])
                gen_noise.loc[gen_filenames[g]] = [ar1.corr, ar1.noise_sd]
                corr_noise = np.zeros((data.shape[0], rand_size), dtype=np.float32)
                corr_noise[0] = rs.normal(size=(1, rand_size))
                for i in range(1, corr_noise.shape[0]):
                    corr_noise[i] = ar1.update(corr_noise[i - 1], rs)
                gen_preds["corr"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[["X_t", "Ux_t"]],
                                                                                     corr_noise, batch_size=batch_size,
                                                                                     stochastic=1)
    return gen_preds, gen_noise


def calc_pdf_kde(x, x_bins, bandwidth=0.5, algorithm="kd_tree", leaf_size=100):
    kde = KernelDensity(bandwidth=bandwidth, algorithm=algorithm, leaf_size=leaf_size)
    kde.fit(x.reshape(-1, 1))
    pdf = np.exp(kde.score_samples(x_bins.reshape(-1, 1)))
    return pdf


def calc_pdf_gmm(x, x_bins, n_components=4):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(x.reshape(-1, 1))
    pdf = np.exp(gmm.score_samples(x_bins.reshape(-1, 1)))
    return pdf


def time_correlations(data, time_lags):
    data_series = pd.Series(data)
    gen_time_corr = np.zeros(time_lags.size, dtype=np.float32)
    for t, time_lag in enumerate(time_lags):
        gen_time_corr[t] = data_series.autocorr(lag=time_lag)
    return gen_time_corr


def run_offline_analysis(gan_index, data, gan_path, seed, batch_size,
                         out_dir, meta_columns,
                         pdf_bins=np.arange(-16, 23, 0.1),
                         bandwidth=0.2,
                         time_lags=np.arange(1, 500), x_index=0):
    epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30],
                      dtype=np.int32)
    print("Calculate offline GAN predictions {0:03d}".format(gan_index))
    gan_preds, gan_noise = offline_gan_predictions(gan_index, data, gan_path, seed=seed, batch_size=batch_size)
    print("Saving offline GAN predictions {0:03d}".format(gan_index))
    pd.merge(data[meta_columns], gan_preds, left_index=True, right_index=True).to_csv(
        join(out_dir, "gan_{0:03d}_offline_predictions.csv".format(gan_index)), index_label="Index")
    pdf_columns = ["truth"]
    for key in gan_preds.keys():
        pdf_columns += [key + "_" + x for x in gan_preds[key].columns]
    pdfs = pd.DataFrame(0.0, index=pdf_bins, columns=pdf_columns, dtype=np.float32)
    print("Calc PDFs of GAN predictions {0:03d}".format(gan_index))
    pdfs.loc[:, "truth"] = calc_pdf_kde(data["Ux_t+1"], pdf_bins, bandwidth=bandwidth)
    pdfs.to_csv(join(out_dir, "gan_{0:03d}_offline_pdfs.csv".format(gan_index)), index_label="Bins")
    for key in gan_preds.keys():
        for col in gan_preds[key].columns:
            pdfs.loc[:, key + "_" + col] = calc_pdf_kde(gan_preds[key][col], pdf_bins, bandwidth=bandwidth)
    print("Calc Hellingers of GAN predictions {0:03d}".format(gan_index))
    hellingers = pd.DataFrame(0.0, index=epochs,
                              columns=["{0:04d}_{1}".format(gan_index, k) for k in gan_preds.keys()],
                              dtype=np.float32)
    for key in gan_preds.keys():
        for c, col in enumerate(gan_preds[key].columns):
            hellingers.loc[epochs[c], "{0:04d}_{1}".format(gan_index, key)] = hellinger(pdfs["truth"],
                                                                                        pdfs[key + "_" + col])
    hellingers.to_csv(join(out_dir, "gan_{0:03d}_offline_hellinger.csv".format(gan_index)), index_label="Epoch")
    print("Calc time correlations of GAN predictions {0:03d}".format(gan_index))
    gan_time_corr = pd.DataFrame(0.0, index=time_lags, columns=pdfs.columns, dtype=np.float32)
    x_points = data["x_index"] == x_index
    for col in gan_time_corr.columns:
        if col == "truth":
            gan_time_corr.loc[:, col] = time_correlations(data.loc[x_points, "Ux_t+1"], time_lags)
        else:
            key = col.split("_")[0]
            mod = col[len(key) + 1]
            gan_time_corr.loc[:, col] = time_correlations(gan_preds[key].loc[x_points, mod], time_lags)

    return