import subprocess

def main():
    config_nums = [0, 300, 301, 302, 303, 402, 403]
    config_types = ["climate", "forecast_20"]
    n_procs = [1, 36]
    for t, config_type in enumerate(config_types):
        for config_num in config_nums:
            script = create_submission_script(config_num, 
                                              forecast_type=config_type,
                                              n_procs=n_procs[t])
            #print(script)
            script_filename = "gan_{0}_{1:03d}.sh".format(config_type, config_num)
            print(script_filename)
            with open(script_filename, "w") as script_file:
                script_file.write(script)
            subprocess.run(["qsub", script_filename])

    return


def create_submission_script(config_num, 
                             config_path="config/exp_20_stoch/",
                             account="P54048000", 
                             walltime="08:00:00",
                             queue="regular",
                             email="dgagne@ucar.edu",
                             forecast_type="climate",
                             n_procs=1,):
    sub_str = "#!/bin/bash\n"
    sub_str += "#PBS -N {0}_gan_{1:03d}\n".format(forecast_type[0], config_num)
    sub_str += "#PBS -A {0}\n".format(account)
    sub_str += "#PBS -q {0}\n".format(queue)
    sub_str += "#PBS -l walltime={0}\n".format(walltime)
    sub_str += "#PBS -l select=1:ncpus=36:ompthreads=36\n"
    sub_str += "#PBS -m abe\n"
    sub_str += "#PBS -M {0}\n".format(email)
    sub_str += "#PBS -j oe\n"
    sub_str += "module unload ncarenv\n"
    sub_str += "source /glade/u/home/dgagne/.bash_profile\n"
    sub_str += "source activate deep\n"
    sub_str += "cd /glade/u/home/dgagne/lorenz_gan\n"
    sub_str += "python -u run_lorenz_forecast.py {0}{1}_gan_n_{2:03d}_c_dense.yaml -p {3:d} &> gan_{2:03d}_{1}.log\n".format(config_path, forecast_type, config_num,
    n_procs)
    return sub_str

if __name__ == "__main__":
    main()
