from keras.models import Model, load_model, save_model
import keras.backend as K
from keras.layers import Dense, Activation, Input, LeakyReLU, Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import rv_histogram, norm
from lorenz_gan.gan import Interpolate1D, unnormalize_data, normalize_data, ConcreteDropout
from sklearn.linear_model import LinearRegression
import yaml

class SubModelGAN(object):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model_path_start = "/".join(model_path.split("/")[:-1])
        self.model_config = model_path.split("/")[-1].split("_")[2]
        self.model = load_model(self.model_path, custom_objects={"Interpolate1D": Interpolate1D,
                                                                 "ConcreteDropout": ConcreteDropout})
        self.pred_func = K.function(self.model.input + [K.learning_phase()], [self.model.output])
        self.x_scaling_file = join(self.model_path_start, "gan_X_scaling_values_{0}.csv".format(self.model_config))
        self.y_scaling_file = join(self.model_path_start, "gan_Y_scaling_values_{0}.csv".format(self.model_config))
        self.y_scaling_values = pd.read_csv(self.y_scaling_file, index_col="Channel")
        self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")

    def predict_percentile(self, cond_x, random_x, num_samples=20):
        norm_x = normalize_data(np.expand_dims(cond_x, axis=2), scaling_values=self.x_scaling_values)[0]
        output = np.zeros((cond_x.shape[0], num_samples))
        predictions = np.zeros((cond_x.shape[0],))
        percentiles = norm.cdf(random_x[:, 0]) * 100
        for s in range(num_samples):
            output[:, s] = unnormalize_data(self.pred_func([norm_x[:, :, 0], random_x, True])[0],
                                            self.y_scaling_values)[:, :, 0].sum(axis=1)
        for p in range(predictions.size):
            predictions[p] = np.percentile(output[p], percentiles[p])
        return predictions

    def predict(self, cond_x, random_x):
        norm_x = normalize_data(np.expand_dims(cond_x, axis=2), scaling_values=self.x_scaling_values)[0]
        predictions = unnormalize_data(self.pred_func([norm_x[:, :, 0], random_x, True])[0],
                                       self.y_scaling_values)[:, :, 0].sum(axis=1)
        return predictions


class SubModelHist(object):
    def __init__(self, num_x_bins=20, num_u_bins=20):
        self.num_x_bins = num_x_bins
        self.num_u_bins = num_u_bins
        self.x_bins = None
        self.u_bins = None
        self.model = None

    def fit(self, cond_x, u):
        x_bins = np.linspace(cond_x.min(), cond_x.max(), self.num_x_bins)
        u_bins = np.linspace(cond_x.min(), cond_x.max(), self.num_u_bins)
        self.model, self.x_bins, self.u_bins = np.histogram2d(cond_x, u, bins=(x_bins, u_bins))

    def predict(self, cond_x, random_x):
        cond_x_filtered = np.where(cond_x > self.x_bins.max(), self.x_bins.max(), cond_x)
        cond_x_filtered = np.where(cond_x < self.x_bins.min(), self.x_bins.min(), cond_x_filtered)
        random_percentile = norm.cdf(random_x)
        sampled_u = np.zeros(cond_x.shape)
        for c, cond_x_val in enumerate(cond_x_filtered):
            x_bin = np.searchsorted(self.x_bins, cond_x_val)
            sampled_u[c] = rv_histogram((self.model[:, x_bin[0]], self.u_bins)).ppf(random_percentile[c])
        return sampled_u.ravel()


class AR1RandomUpdater(object):
    def __init__(self, corr=0, noise_sd=1):
        self.corr = corr
        self.noise_sd = noise_sd

    def fit(self, data):
        self.corr = np.corrcoef(data[:-1], data[1:])[0, 1]
        self.noise_sd = np.sqrt(1 - self.corr ** 2)

    def update(self, random_values, rs=None):
        return self.corr * random_values + norm.rvs(size=random_values.shape,
                                                    loc=0, scale=self.noise_sd, 
                                                    random_state=rs)


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
        elif self.noise_type == "multiplicative":
            sampled_u = (1 + random_x) * sampled_u
        return sampled_u.ravel()


class SubModelPolyAdd(object):
    def __init__(self, num_terms=3):
        self.num_terms = num_terms
        self.model = LinearRegression()
        self.res_sd = 0
        self.corr = 1

    def fit(self, x_col, u):
        x_terms = np.zeros((x_col.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = x_col ** p
        self.model.fit(x_terms, u)
        u_mean = self.model.predict(x_terms)
        residuals = u - u_mean
        self.corr = np.corrcoef(residuals[1:], residuals[:-1])[0, 1]
        self.res_sd = np.std(residuals)

    def predict(self, x, residuals=None):
        if residuals is None:
            residuals = np.zeros(x.shape[0])
        x_terms = np.zeros((x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = x ** p
        u_mean = self.model.predict(x_terms)
        u_res = self.corr * residuals + \
            self.res_sd * np.sqrt(1 - self.corr ** 2) * np.random.normal(size=residuals.shape)
        return u_mean, u_res


class SubModelANN(object):
    def __init__(self, inputs=1, hidden_layers=2, hidden_neurons=8,
                 activation="selu", l2_weight=0.01, learning_rate=0.001, loss="mse",
                 noise_sd=1, beta_1=0.9, model_path=None, dropout_alpha=0.5,
                 num_epochs=10, batch_size=1024, verbose=0, model_config=0):
        self.config = dict(inputs=inputs,
                           hidden_layers=hidden_layers,
                           hidden_neurons=hidden_neurons,
                           activation=activation,
                           l2_weight=l2_weight,
                           learning_rate=learning_rate,
                           loss=loss,
                           noise_sd=noise_sd,
                           beta_1=beta_1,
                           dropout_alpha=dropout_alpha,
                           model_path=model_path,
                           num_epochs=num_epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           model_config=model_config)
        if model_path is None:
            nn_input = Input((inputs, ))
            nn_model = nn_input
            for h in range(hidden_layers):
                nn_model = Dense(hidden_neurons, kernel_regularizer=l2(l2_weight))(nn_model)
                if activation == "leaky":
                    nn_model = LeakyReLU(0.1)(nn_model)
                else:
                    nn_model = Activation(activation)(nn_model)
                nn_model = Dropout(dropout_alpha)(nn_model)
                nn_model = GaussianNoise(noise_sd)(nn_model)
            nn_model = Dense(1)(nn_model)
            self.model = Model(nn_input, nn_model)
            self.model.compile(Adam(lr=learning_rate, beta_1=beta_1), loss=loss)
            self.x_scaling_file = None
            self.x_scaling_values = None
        elif type(model_path) == str:
            self.model = load_model(model_path)
            model_path_start = join(model_path.split("/")[:-1])
            self.x_scaling_file = join(model_path_start, "ann_config_{0}_scale.csv".format(self.config["model_config"]))
            self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")

    def fit(self, cond_x, u):
        norm_x, self.x_scaling_values = normalize_data(cond_x,
                                                       scaling_values=self.x_scaling_values)
        self.model.fit(norm_x, u, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"],
                       verbose=self.config["verbose"])

    def predict(self, cond_x, residuals=None):
        norm_x = normalize_data(cond_x, scaling_values=self.x_scaling_values)[0]
        sample_predict = K.function([self.model.input, K.learning_phase()], [self.model.output])
        u_mean = sample_predict([norm_x, 0])[0].ravel()
        u_total = sample_predict([norm_x, 1])[0].ravel()
        u_res = u_total - u_mean
        return u_mean, u_res

    def save_model(self, out_path):
        out_config_file = join(out_path, "ann_res_config_{0:04d}_opts.yaml".format(self.config["model_config"]))
        with open(out_config_file, "w") as out_config:
            yaml.dump(self.config, out_config)
        model_file = join(out_path, "ann_res_config_{0:04d}_model.nc".format(self.config["model_config"]))
        save_model(self.model, model_file)
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")


class SubModelANNRes(object):
    def __init__(self, mean_inputs=1, res_inputs=2, hidden_layers=2, hidden_neurons=8,
                 activation="selu", l2_weight=0.01, learning_rate=0.001, loss="mse",
                 noise_sd=1, beta_1=0.9, model_path=None, dropout_alpha=0.5,
                 num_epochs=10, batch_size=1024, verbose=0, model_config=0):
        self.config = dict(mean_inputs=mean_inputs,
                           res_inputs=res_inputs,
                           hidden_layers=hidden_layers,
                           hidden_neurons=hidden_neurons,
                           activation=activation,
                           l2_weight=l2_weight,
                           learning_rate=learning_rate,
                           loss=loss,
                           noise_sd=noise_sd,
                           beta_1=beta_1,
                           dropout_alpha=dropout_alpha,
                           model_path=model_path,
                           num_epochs=num_epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           model_config=model_config)
        if model_path is None:
            nn_input = Input((mean_inputs,))
            nn_model = nn_input
            for h in range(hidden_layers):
                nn_model = Dense(hidden_neurons, kernel_regularizer=l2(l2_weight))(nn_model)
                if activation == "leaky":
                    nn_model = LeakyReLU(0.1)(nn_model)
                else:
                    nn_model = Activation(activation)(nn_model)
            nn_model = Dense(1)(nn_model)
            nn_res_input = Input((res_inputs,))
            nn_res = nn_res_input
            for h in range(hidden_layers):
                nn_res = Dense(hidden_neurons, kernel_regularizer=l2(l2_weight))(nn_res)
                if activation == "leaky":
                    nn_res = LeakyReLU(0.1)(nn_res)
                else:
                    nn_res = Activation(activation)(nn_res)
                nn_res = Dropout(dropout_alpha)(nn_res)
                nn_res = GaussianNoise(noise_sd)(nn_res)
            nn_res = Dense(1)(nn_res)
            self.mean_model = Model(nn_input, nn_model)
            self.mean_model.compile(Adam(lr=learning_rate, beta_1=beta_1), loss=loss)
            self.res_model = Model(nn_res_input, nn_res)
            self.res_model.compile(Adam(lr=learning_rate, beta_1=beta_1), loss=loss)
            self.x_scaling_file = None
            self.x_scaling_values = None
        elif type(model_path) == str:
            mean_model_file = join(model_path, "ann_res_config_{0:04d}_mean.nc".format(self.config["model_config"]))
            res_model_file = join(model_path, "ann_res_config_{0:04d}_res.nc".format(self.config["model_config"]))
            self.mean_model = load_model(mean_model_file)
            self.res_model = load_model(res_model_file)
            self.x_scaling_file = join(model_path, "gan_X_scaling_values_{0}.csv".format(model_config))
            self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")
        self.res_predict = K.function([self.res_model.input, K.learning_phase()], [self.res_model.output])

    def fit(self, cond_x, u):
        norm_x, self.x_scaling_values = normalize_data(cond_x,
                                                       scaling_values=self.x_scaling_values)
        print(self.x_scaling_values)
        self.mean_model.fit(norm_x, u, batch_size=self.config["batch_size"],
                            epochs=self.config["num_epochs"],
                            verbose=self.config["verbose"])
        mean_preds = self.mean_model.predict(norm_x).ravel()
        residuals = u - mean_preds
        res_input = np.vstack([mean_preds[:-1], residuals[:-1]]).T
        self.res_model.fit(res_input, residuals[1:], batch_size=self.config["batch_size"],
                            epochs=self.config["num_epochs"],
                            verbose=self.config["verbose"])

    def predict(self, cond_x, residuals=None):
        norm_x = normalize_data(cond_x, scaling_values=self.x_scaling_values)[0]
        u_mean = self.mean_model.predict(norm_x).ravel()
        if residuals is None:
            res_input = u_mean.reshape(-1, 1)
        else:
            res_input = np.zeros((u_mean.shape[0], 2))
            res_input[:, 0] = u_mean
            res_input[:, 1] = residuals
        u_res = self.res_predict([res_input, 1])[0].ravel()
        return u_mean, u_res

    def save_model(self, out_path):
        out_config_file = join(out_path, "ann_res_config_{0:04d}_opts.yaml".format(self.config["model_config"]))
        with open(out_config_file, "w") as out_config:
            yaml.dump(self.config, out_config)
        mean_model_file = join(out_path, "ann_res_config_{0:04d}_mean.nc".format(self.config["model_config"]))
        res_model_file = join(out_path, "ann_res_config_{0:04d}_res.nc".format(self.config["model_config"]))
        save_model(self.mean_model, mean_model_file)
        save_model(self.res_model, res_model_file)
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")


def load_ann_model(model_path, model_type, model_config):
    config_filename = join(model_path, "{0}_config_{1:04d}_opts.yaml".format(model_type, model_config))
    with open(config_filename) as config_file:
        config = yaml.load(config_file)
    if model_type == "ann":
        model = SubModelANN(**config)
    else:
        model = SubModelANNRes(**config)
    return model
