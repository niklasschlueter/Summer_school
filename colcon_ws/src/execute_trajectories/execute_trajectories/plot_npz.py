import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_npz(filename):
    data = np.load(filename)

    # list available arrays
    print("Arrays in file:", list(data.keys()))

    time = data["time"]

    # Plot each array except time
    for key in data.files:
        if key == "time":
            continue

        arr = data[key]

        # Handle 1D vs 2D
        if arr.ndim == 1:
            plt.figure()
            plt.plot(time, arr)
            plt.title(key)
            plt.xlabel("time [s]")
            plt.ylabel(key)
        else:
            plt.figure()
            for i in range(arr.shape[1]):
                plt.plot(time, arr[:, i], label=f"{key}[{i}]")
            plt.title(key)
            plt.xlabel("time [s]")
            plt.ylabel(key)
            plt.legend()

    plt.show()



def plot_multiple_runs(pattern="run_*.npz", alpha=0.2, normalize_time=True):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files")

    # Load first file just to know what arrays exist
    sample = np.load(files[0])
    arrays = list(sample.keys())

    for key in arrays:
        if key == "time":
            continue

        plt.figure()
        for fname in files:
            data = np.load(fname)
            time = data["time"]

            # normalize each run’s time so it starts at zero
            if normalize_time:
                time = time - time[0]

            arr = data[key]

            if arr.ndim == 1:
                plt.plot(time, arr, alpha=alpha)
            else:
                for i in range(arr.shape[1]):
                    plt.plot(time, arr[:, i], alpha=alpha)

        plt.title(f"{key} across {len(files)} runs")
        plt.xlabel("time [s]")
        plt.ylabel(key)

    plt.show()



if __name__ == "__main__":
    #plot_npz("runs/run_0.npz")
    plot_multiple_runs("runs_5/run_*.npz", alpha=0.2)
