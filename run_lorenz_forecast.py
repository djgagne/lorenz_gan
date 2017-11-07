import numpy as np
import pandas as pd
import yaml
import argparse
from lorenz_gan.lorenz import run_lorenz96_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="Path to config yaml file")
    args = parser.parse_args()

    return


if __name__ == "__main__":
    main()