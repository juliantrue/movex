import matplotlib.pyplot as plt
import pandas as pd


def main():
    path_to_results = "results/mot20_eval/baseline/results.csv"

    df = pd.read_csv(path_to_results)

    plot_AP_vs_od_latency(df)
    plot_avg_latency_vs_od_latency(df)


def plot_AP_vs_od_latency(df):
    plt.figure(figsize=(8, 7))
    plt.plot(df["object_detection_latency_(ms)"], df["AP"], "bo-")
    plt.title("AP vs Simulated Object Detector Latency")
    plt.xlabel("Simulated Object Detector Latency (ms)")
    plt.ylabel("AP")
    plt.grid()
    plot_name = "OD_latency_v_AP.png"
    print(f"Saving plot: {plot_name}")
    plt.savefig(plot_name)


def plot_avg_latency_vs_od_latency(df):
    plt.figure(figsize=(8, 7))
    plt.plot(df["object_detection_latency_(ms)"], df["average_latency(ms)"], "bo-")
    plt.title("Average Computation Time Per Frame vs Simulated Object Detector Latency")
    plt.xlabel("Simulated Object Detector Latency (ms)")
    plt.ylabel("Average Computation Time Per Frame (ms)")
    plt.grid()

    plot_name = "avg_latency_v_OD_latency.png"
    print(f"Saving plot: {plot_name}")
    plt.savefig(plot_name)


if __name__ == "__main__":
    main()
