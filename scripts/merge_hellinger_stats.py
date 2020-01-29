import pandas as pd
import numpy as np
from os.path import join
from glob import glob

def main():
    offline_path = "/glade/work/dgagne/exp_20_stoch/offline_20190409/"
    hellinger_files = pd.Series(sorted(glob(join(offline_path, "*hellinger.csv"))))
    gan_number = hellinger_files.str.split("/").str[-1].str.split("_").str[1]
    hellinger_score = np.zeros(gan_number.shape)
    for h, hf in enumerate(hellinger_files):
        print(hf)
        hell_data = pd.read_csv(hf)
        hellinger_score[h] = hell_data.iloc[-1, 1]
    hell_df = pd.DataFrame({"gan": gan_number, "hellinger_offline": hellinger_score})
    print(hell_df)
    hell_df.to_csv("merged_offline_hellinger_scores.csv")
    return

if __name__ == "__main__":
    main()
