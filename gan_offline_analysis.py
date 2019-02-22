import argparse
import yaml
from dask.distributed import Client, LocalCluster, wait
from lorenz_gan.analysis import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="lorenz.yaml", help="Config yaml file")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    cluster = LocalCluster(n_workers=args.nprocs)
    client = Client(cluster)
    gan_indices = config["gan_indices"]
    data_file = config["data_file"]
    all_data = pd.read_csv(data_file)
    data = client.scatter(all_data)
    jobs = []
    for gan_index in gan_indices:
        jobs.append(client.submit(run_offline_analysis, gan_index, data, config["gan_path"],
                                  config["seed"], config["batch_size"], config["meta_columns"]))
    wait(jobs)
    return


if __name__ =="__main__":
    main()