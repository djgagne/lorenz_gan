from keras.models import load_model
import pickle
import numpy as np
from scipy.stats import rv_histogram, norm


class SubModelGAN(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def predict(self, cond_x, random_x):
        return self.model.predict([cond_x, random_x]).sum(axis=1)


class SubModelHist(object):
    def __init__(self, num_x_bins, num_u_bins):
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
            sampled_u[c] = rv_histogram(self.histogram[x_bin]).ppf(random_percentile[c])
        return sampled_u


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

