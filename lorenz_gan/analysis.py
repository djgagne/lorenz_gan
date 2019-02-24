import pandas as pd
import numpy as np
from lorenz_gan.submodels import SubModelGAN, AR1RandomUpdater
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from glob import glob
from os.path import join
import keras.backend as K
import traceback

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
    random_values = rs.normal(size=(data.shape[0], rand_size))
    all_zeros = np.zeros((data.shape[0], rand_size), dtype=np.float32)
    corr_noise = np.zeros((data.shape[0], rand_size), dtype=np.float32)
    gen_preds = dict()
    for pred_type in ["det", "rand", "corr"]:
        gen_preds[pred_type] = pd.DataFrame(0, index=data.index, columns=gen_filenames,
                             dtype=np.float32)
    gen_noise = pd.DataFrame(0.0, dtype=np.float32, index=gen_filenames, columns=["corr", "noise_sd"])
    for g, gen_file in enumerate(gen_files):
        sess = K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1,
                                                inter_op_parallelism_threads=1,
                                                gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(sess)
        K.tf.set_random_seed(seed)
        with K.tf.device("/cpu:0"):
            print("Predicting " + gen_filenames[g])
            gen_model = SubModelGAN(gen_file)
            if gen_model.x_scaling_values.shape[0] == 1:
                input_cols = ["X_t"]
            else:
                input_cols = ["X_t", "Ux_t"]
            print(gen_filenames[g], "Det preds")
            gen_preds["det"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[input_cols],
                                                                                all_zeros, batch_size=batch_size,
                                                                                stochastic=0)
            print(gen_filenames[g], "Rand preds")
            gen_preds["rand"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[input_cols],
                                                                                    random_values,
                                                                                    batch_size=batch_size,
                                                                                    stochastic=1)
            print(gen_filenames[g], "Random updater")
            ar1 = AR1RandomUpdater()
            x_indices = data["x_index"] == 0
            ar1.fit(data.loc[x_indices, "Ux_t+1"] - gen_preds["det"].loc[x_indices, gen_filenames[g]])
            print(gen_filenames[g], ar1.corr, ar1.noise_sd)
            gen_noise.loc[gen_filenames[g]] = [ar1.corr, ar1.noise_sd]
            rs_corr = np.random.RandomState(seed)
            corr_noise[0] = rs_corr.normal(size=(1, rand_size))
            print(gen_filenames[g], "random noise")
            for i in range(1, corr_noise.shape[0]):
                corr_noise[i] = ar1.update(corr_noise[i - 1], rs_corr)
            print(gen_filenames[g], "Corr preds")
            gen_preds["corr"].loc[:, gen_filenames[g]] = gen_model.predict_batch(data[input_cols],
                                                                                    corr_noise, batch_size=batch_size,
                                                                                    stochastic=1)
            del ar1
            del gen_model.model
            del gen_model
            corr_noise[:] = 0
        sess.close()
        del sess
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



