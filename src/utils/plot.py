import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_from_tensorboard(log_dir, tag="reward/mean", save_path="reward_curve.png"):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values)
    plt.xlabel("Iteration")
    plt.ylabel(tag)
    plt.title(f"Training: {tag}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    plot_from_tensorboard(log_dir)