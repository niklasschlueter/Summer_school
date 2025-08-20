import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    plot_npz("runs/run_0.npz")
