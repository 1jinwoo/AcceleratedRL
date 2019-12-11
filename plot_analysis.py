import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    import datetime
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date

def append_or_create_list_for_key(dict, key, ele):
    if key in dict:
        dict[key].append(ele)
    else:
        dict[key] = [ele]

def smooth(scalars, weight=0.5):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def running_avg(list_to_avg, avg_steps=100):
    array_to_avg = np.asarray(list_to_avg)
    array_to_avg = array_to_avg.reshape(array_to_avg.shape[0], -1)
    array_cum_sum = np.copy(array_to_avg)
    for i in range(1, array_to_avg.shape[1]):
        array_cum_sum[:, i] = array_cum_sum[:, i - 1] + array_to_avg[:, i]

    array_avged = (array_cum_sum[:, avg_steps:] - array_cum_sum[:, :-avg_steps]) / avg_steps
    # array_avged = smooth(array_avged)

    return array_avged

if __name__ == "__main__":
    rl_avg_steps = 1000
    metrics_to_plot = ["reward"]
    parent_dir = os.path.join("results", "pitfall_ppo2_rl_justin")
    plot_dir = os.path.join(parent_dir, "plots")
    plot_dir = addDateTime(plot_dir)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    performance_dict = {}
    dirname = parent_dir
    performance_file = os.path.join(dirname, "performance.p")
    model_dir = os.path.join(dirname, "models")
    if not os.path.exists(performance_file):
        print(performance_file)

    with open(performance_file, "rb") as f:
        performance = pickle.load(f)
        append_or_create_list_for_key(performance_dict, "rl_baseline", performance)

    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        for key, val_old in performance_dict.items():
            val = running_avg([ele[metric] for ele in val_old], rl_avg_steps)
            val_mean = val.mean(axis=0)
            val_std = val.std(axis=0)
            line, = plt.plot(val_mean, label=key)
            plt.fill_between(range(0, len(val_mean)),
                             val_mean + val_std, val_mean - val_std, alpha=0.2)

            plt.legend()
            plt.ylabel(metric)
            plt.xlabel("steps")
            # plt.xlim((0, len(val_mean)))
            plt.xlim((0, len(val_mean)))
            plt.minorticks_on()
            plt.grid(True, which="both", alpha=.2)
            plt.savefig(os.path.join(plot_dir, "{}.png".format(metric)))
            plt.show()