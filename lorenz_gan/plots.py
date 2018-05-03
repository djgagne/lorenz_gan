import matplotlib.pyplot as plt
import numpy as np
from os.path import join


def plot_forcing_histogram(truth_forcing, model_forcing, out_path, bins=np.arange(-30, 35, 5), **kwargs):
    """
    Plot 1D histogram of subgrid forcing versus the frequency of each forcing value. The Truth model forcing
    is shown in solid fill while the model forcing is shown as a thick line on top of the truth histogram.

    Args:
        truth_forcing: Array of truth model U values
        model_forcing: Array of GAN or other parameterization forcing values.
        out_path: Path to where figure is output
        bins: Array of bin edge values
        **kwargs: Arguments to modify default figure values. Other kwargs are ignored.

    Returns:

    """
    defaults = dict(figsize=(6, 4),
                    fontsize=14,
                    truth_color="gray",
                    truth_label="Truth",
                    model_color="red",
                    model_label="GAN",
                    title="Forcing Distribution",
                    xlabel="Subgrid Forcing $U_{t+1}$",
                    ylabel="Frequency",
                    lw=2,
                    image_type="png",
                    dpi=200,
                    bbox_inches="tight",
                    auto_close=False)
    for kw_key in kwargs.keys():
        defaults[kw_key] = kwargs[kw_key]
    plt.figure(figsize=defaults["figsize"])
    plt.hist(truth_forcing, bins=bins, color=defaults["truth_color"], label=defaults["truth_label"])
    plt.hist(model_forcing, bins=bins, histtype='step', lw=defaults["lw"], label=defaults["model_label"],
             color=defaults["model_color"])
    plt.legend(loc=0, fontsize=defaults["fontsize"])
    plt.xlabel(defaults["xlabel"], fontsize=defaults["fontsize"])
    plt.ylabel(defaults["yabel"], fontsize=defaults["fontsize"])
    plt.title(defaults["title"], fontsize=defaults["fontsize"] + 2)
    plt.savefig(join(out_path, "forcing_dist_hist_{0}.{1}".format(defaults["model_label"].lower(),
                                                                  defaults["image_type"])),
                dpi=defaults["dpi"],
                bbox_inches=defaults["bbox_inches"])
    if defaults["auto_close"]:
        plt.close()
