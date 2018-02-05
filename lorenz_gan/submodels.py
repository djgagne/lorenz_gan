from keras.models import load_model
import keras.backend as K
import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import rv_histogram, norm
from lorenz_gan.gan import Interpolate1D, unnormalize_data
from sklearn.linear_model import LinearRegression


class SubModelGAN(object):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model_path_start = "/".join(model_path.split("/")[:-1])
        self.model_config = model_path.split("/")[-1].split("_")[2]
        self.model = load_model(self.model_path, custom_objects={"Interpolate1D": Interpolate1D})
        self.pred_func = K.function(self.model.input + [K.learning_phase()], [self.model.output])
        self.scaling_file = join(self.model_path_start, "gan_Y_scaling_values_{0}.csv".format(self.model_config))
        self.scaling_values = pd.read_csv(self.scaling_file, index_col="Channel")

    def predict(self, cond_x, random_x):
        predictions = unnormalize_data(self.pred_func([cond_x, random_x, True])[0],
                                       self.scaling_values)[:, :, 0].sum(axis=1)
        return predictions


class SubModelHist(object):
    def __init__(self, num_x_bins=20, num_u_bins=20):
        self.num_x_bins = num_x_bins
        self.num_u_bins = num_u_bins
        self.x_bins = None
        self.u_bins = None
        self.histogram = None

    def fit(self, cond_x, u):
        x_bins = np.linspace(cond_x.min(), cond_x.max(), self.num_x_bins)
        u_bins = np.linspace(cond_x.min(), cond_x.max(), self.num_u_bins)
        self.histogram, self.x_bins, self.u_bins = np.histogram2d(cond_x, u, bins=(x_bins, u_bins))

    def predict(self, cond_x, random_x):
        cond_x_filtered = np.where(cond_x > self.x_bins.max(), self.x_bins.max(), cond_x)
        cond_x_filtered = np.where(cond_x < self.x_bins.min(), self.x_bins.min(), cond_x_filtered)
        random_percentile = norm.cdf(random_x)
        sampled_u = np.zeros(cond_x.shape)
        for c, cond_x_val in enumerate(cond_x_filtered):
            x_bin = np.searchsorted(self.x_bins, cond_x_val)
            sampled_u[c] = rv_histogram((self.histogram[:, x_bin[0]], self.u_bins)).ppf(random_percentile[c])
        return sampled_u.ravel()


class AR1RandomUpdater(object):
    def __init__(self, corr=0, noise_sd=1):
        self.corr = corr
        self.noise_sd = noise_sd

    def fit(self, data):
        self.corr = np.corrcoef(data[:-1], data[1:])[0, 1]
        self.noise_sd = np.sqrt(1 - self.corr ** 2)

    def update(self, random_values):
        return self.corr * random_values + norm.rvs(size=random_values.shape,
                                                    loc=0, scale=self.noise_sd)


class SubModelPoly(object):
    def __init__(self, num_terms=3, noise_type="additive"):
        self.num_terms = num_terms
        self.model = LinearRegression()
        self.noise_type = noise_type

    def fit(self, cond_x, u):
        x_terms = np.zeros((cond_x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = cond_x ** p
        self.model.fit(x_terms, u)

    def predict(self, cond_x, random_x):
        x_terms = np.zeros((cond_x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1:p] = cond_x ** p
        sampled_u = self.model.predict(x_terms).reshape(cond_x.shape)
        if self.noise_type == "additive":
            sampled_u += random_x
        if self.noise_type == "multiplicative":
            sampled_u = (1 + random_x) * sampled_u
        return sampled_u.ravel()

