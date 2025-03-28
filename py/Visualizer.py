import numpy as np
import matplotlib.pyplot as plt
import HMatrix as hm


def plot_compression_metrics(metrics: list[dict]):
    """Plot the compression metrics."""
    sizes = [
        [m["original_size"], m["compressed_size"], m["dimensions"]] for m in metrics
    ]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(
        x - width / 2,
        [s[0] for s in sizes],
        width,
        label="Original Size",
        color="blue",
    )
    plt.bar(
        x + width / 2,
        [s[1] for s in sizes],
        width,
        label="Compressed Size",
        color="orange",
    )

    plt.xlabel("Matrix Size")
    plt.ylabel("Size (bytes)")
    plt.title("Compression Metrics")
    plt.xticks(x, [f"{s[2]}" for s in sizes])
    plt.legend()
    plt.tight_layout()
    plt.savefig("compression_metrics.png")
    plt.show()


if __name__ == "__main__":

    sizes = [2**n for n in range(9, 13)]

    metrics = []

    for size in sizes:
        mat = np.random.rand(size, size)

        hmatrix = hm.HMatrix(mat, max_rank=10, min_size=4, tol=1e-5)
        cur_metrics = hm.measure_compression(hmatrix)
        del hmatrix
        cur_metrics["dimensions"] = f"{size} x {size}"
        metrics.append(cur_metrics)

    plot_compression_metrics(metrics)
