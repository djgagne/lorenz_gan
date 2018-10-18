from keras.models import Model, load_model, save_model
import keras.backend as K
from keras.layers import Dense, Activation, Input, LeakyReLU, Dropout, GaussianNoise, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
from os.path import join, exists
import pandas as pd
from scipy.stats import rv_histogram, norm
from lorenz_gan.gan import Interpolate1D, unnormalize_data, normalize_data, ConcreteDropout, Split1D, Scale
from sklearn.linear_model import LinearRegression
import yaml


class SubModelGAN(object):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model_path_start = "/".join(model_path.split("/")[:-1])
        self.model_config = model_path.split("/")[-1].split("_")[2]
        self.model = load_model(self.model_path, custom_objects={"Interpolate1D": Interpolate1D,
                                                                 "ConcreteDropout": ConcreteDropout,
                                                                 "Split1D": Split1D,
                                                                 "Scale": Scale})
        self.pred_func = K.function(self.model.input + [K.learning_phase()], [self.model.output])
        self.x_scaling_file = join(self.model_path_start, "gan_X_scaling_values_{0}.csv".format(self.model_config))
        self.y_scaling_file = join(self.model_path_start, "gan_Y_scaling_values_{0}.csv".format(self.model_config))
        self.y_scaling_values = pd.read_csv(self.y_scaling_file, index_col="Channel")
        self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")

    def predict(self, cond_x, random_x):
        norm_x = normalize_data(np.expand_dims(cond_x, axis=2), scaling_values=self.x_scaling_values)[0]
        predictions = unnormalize_data(self.pred_func([norm_x[:, :, 0], random_x, True])[0],
                                       self.y_scaling_values)[:, :, 0]
        if predictions.shape[1] > 1:
            predictions = predictions.sum(axis=1)
        else:
            predictions = predictions.ravel()
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

    def fit(self, x, u):
        x_terms = np.zeros((x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = x[:, 0] ** p
        self.model.fit(x_terms, u)
        u_mean = self.model.predict(x_terms)
        residuals = u - u_mean
        self.corr = np.corrcoef(residuals[1:], residuals[:-1])[0, 1]
        self.res_sd = np.std(residuals)
        print("Poly Add Corr:", self.corr)
        print("Poly Add Res SD:", self.res_sd)

    def predict(self, x, residuals=None, predict_residuals=True):
        if residuals is None:
            residuals = np.zeros(x.shape[0])
        x_terms = np.zeros((x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = x[:, 0] ** p
        u_mean = self.model.predict(x_terms).ravel()
        u_res = self.corr * residuals + \
            self.res_sd * np.sqrt(1 - self.corr ** 2) * np.random.normal(size=residuals.shape)
        u_res = u_res.ravel()
        if predict_residuals:
            return u_mean, u_res
        else:
            return u_mean

    def predict_mean(self, x):
        x_terms = np.zeros((x.shape[0], self.num_terms))
        for p in range(1, self.num_terms + 1):
            x_terms[:, p - 1] = x[:, 0] ** p
        return self.model.predict(x_terms).ravel()

    def predict_res(self, residuals):
        u_res = self.corr * residuals + \
                self.res_sd * np.sqrt(1 - self.corr ** 2) * np.random.normal(size=residuals.shape)
        u_res = u_res.ravel()
        return u_res


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
        model_path_start = join(model_path.split("/")[:-1])
        self.x_scaling_file = join(model_path_start, "ann_config_{0}_scale.csv".format(self.config["model_config"]))
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
            self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")

    def fit(self, cond_x, u):
        norm_x, self.x_scaling_values = normalize_data(cond_x,
                                                       scaling_values=self.x_scaling_values)
        self.model.fit(norm_x, u, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"],
                       verbose=self.config["verbose"])
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")

    def predict(self, cond_x, residuals=None, predict_residuals=True):
        norm_x = normalize_data(cond_x, scaling_values=self.x_scaling_values)[0]
        sample_predict = K.function([self.model.input, K.learning_phase()], [self.model.output])
        u_mean = sample_predict([norm_x, 0])[0].ravel()
        u_total = sample_predict([norm_x, 1])[0].ravel()
        u_res = u_total - u_mean
        if predict_residuals:
            return u_mean, u_res
        else:
            return u_mean

    def save_model(self, out_path):
        out_config_file = join(out_path, "ann_res_config_{0:04d}_opts.yaml".format(self.config["model_config"]))
        with open(out_config_file, "w") as out_config:
            yaml.dump(self.config, out_config)
        model_file = join(out_path, "ann_res_config_{0:04d}_model.nc".format(self.config["model_config"]))
        save_model(self.model, model_file)
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")


class SubModelANNRes(object):
    """
    Artificial Neural Network Parameterization with separate mean and residual models.

    Args:
        mean_inputs (int): number of inputs to mean model (default 1)
        hidden_layers (int): number of hidden layers in each model (default 2)
        hidden_neurons (int): number of hidden neurons in each hidden layer (default 8)
        noise_sd (float): standard deviation of the GaussianNoise layers (default 1)
        beta_1 (float): controls the beta_1 parameter in the Adam optimizer
        model_path (str or None): Path to existing model object. If not specified or if model file not found,
            new model is created from scratch.
        dropout_alpha (float): Proportion of input neurons set to 0.
        num_epochs (int): The number of epochs (iterations through training data) performed during training.
        batch_size (int): Number of training examples sampled for each network update
        val_split (float): Proportion of training examples used to split training and validation data
        verbose (int): Level of text output during training.
        model_config (int): Configuration number to keep saved files consistent.

    """
    def __init__(self, mean_inputs=1, hidden_layers=2, hidden_neurons=8,
                 activation="selu", l2_weight=0.01, learning_rate=0.001, mean_loss="mse",
                 res_loss="kullback_leibler_divergence",
                 noise_sd=1, beta_1=0.9, model_path=None, dropout_alpha=0.5,
                 num_epochs=10, batch_size=1024, val_split=0.5, verbose=0, model_config=0):
        self.config = dict(mean_inputs=mean_inputs,
                           hidden_layers=hidden_layers,
                           hidden_neurons=hidden_neurons,
                           activation=activation,
                           l2_weight=l2_weight,
                           learning_rate=learning_rate,
                           mean_loss=mean_loss,
                           res_loss=res_loss,
                           noise_sd=noise_sd,
                           beta_1=beta_1,
                           dropout_alpha=dropout_alpha,
                           model_path=model_path,
                           num_epochs=num_epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           model_config=model_config,
                           val_split=val_split)
        mean_model_file = join(model_path, "annres_config_{0:04d}_mean.nc".format(self.config["model_config"]))
        res_model_file = join(model_path, "annres_config_{0:04d}_res.nc".format(self.config["model_config"]))
        self.x_scaling_file = join(model_path, "annres_scaling_values_{0:04d}.csv".format(model_config))
        if model_path is None or not exists(mean_model_file):
            nn_input = Input((mean_inputs,))
            nn_model = nn_input
            for h in range(hidden_layers):
                nn_model = Dense(hidden_neurons, kernel_regularizer=l2(l2_weight))(nn_model)
                if activation == "leaky":
                    nn_model = LeakyReLU(0.1)(nn_model)
                else:
                    nn_model = Activation(activation)(nn_model)
            nn_model = Dense(1)(nn_model)
            nn_res_input_x = Input((mean_inputs,))
            nn_res_input_res = Input((1,))
            nn_res = concatenate([nn_res_input_x, nn_res_input_res])
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
            self.mean_model.compile(Adam(lr=learning_rate, beta_1=beta_1), loss=mean_loss)
            self.res_model = Model([nn_res_input_x, nn_res_input_res], nn_res)
            self.res_model.compile(Adam(lr=learning_rate, beta_1=beta_1), loss=res_loss)
            self.x_scaling_values = None
        elif type(model_path) == str:
            self.mean_model = load_model(mean_model_file)
            self.res_model = load_model(res_model_file)
            self.x_scaling_values = pd.read_csv(self.x_scaling_file, index_col="Channel")
        self.res_predict = K.function(self.res_model.input + [K.learning_phase()], [self.res_model.output])

    def fit(self, cond_x, u):
        split_index = int(cond_x.shape[0] * self.config["val_split"])
        norm_x, self.x_scaling_values = normalize_data(cond_x,
                                                       scaling_values=self.x_scaling_values)
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")
        self.mean_model.fit(norm_x[:split_index], u[:split_index], batch_size=self.config["batch_size"],
                            epochs=self.config["num_epochs"],
                            verbose=self.config["verbose"])
        mean_preds = self.mean_model.predict(norm_x[split_index:]).ravel()
        residuals = u[split_index:] - mean_preds
        self.res_model.fit([norm_x[split_index:-1], residuals[:-1].reshape(-1, 1)],
                           residuals[1:], batch_size=self.config["batch_size"],
                           epochs=self.config["num_epochs"],
                           verbose=self.config["verbose"])

    def predict(self, cond_x, residuals, predict_residuals=True):
        norm_x = normalize_data(cond_x, scaling_values=self.x_scaling_values)[0]
        u_mean = self.mean_model.predict(norm_x).ravel()
        u_res = self.res_predict([norm_x, residuals, 1])[0].ravel()
        if predict_residuals:
            return u_mean, u_res
        else:
            return u_mean

    def save_model(self, out_path):
        out_config_file = join(out_path, "annres_config_{0:04d}_opts.yaml".format(self.config["model_config"]))
        with open(out_config_file, "w") as out_config:
            yaml.dump(self.config, out_config)
        mean_model_file = join(out_path, "annres_config_{0:04d}_mean.nc".format(self.config["model_config"]))
        res_model_file = join(out_path, "annres_config_{0:04d}_res.nc".format(self.config["model_config"]))
        save_model(self.mean_model, mean_model_file)
        save_model(self.res_model, res_model_file)
        self.x_scaling_values.to_csv(self.x_scaling_file, index_label="Channel")


def load_ann_model(model_config_file):
    """
    Load Artificial Neural Network model from config yaml file

    Args:
        model_config_file: The full or relative path to the config file with name formatted "annres_config_0000.yaml"

    Returns:
        SubModelANN or SubModelANNRes
    """
    model_type = model_config_file.split("/")[-1].split("_")[0]
    with open(model_config_file) as config_file:
        config = yaml.load(config_file)
    if model_type == "ann":
        model = SubModelANN(**config)
    else:
        model = SubModelANNRes(**config)
    return model
