import os
import numpy as np
import matplotlib.pyplot as plt


def produce_trunc_mean_cov(input_signal, input_type, E_val):
    """
    Compute type-specific sample mean and covariance matrices.
    """
    length_per_electrode = input_signal.shape[1] // E_val
    signal_tar = input_signal[input_type == 1, :]
    signal_ntar = input_signal[input_type == -1, :]

    signal_tar_mean = signal_tar.mean(axis=0).reshape(E_val, length_per_electrode)
    signal_ntar_mean = signal_ntar.mean(axis=0).reshape(E_val, length_per_electrode)

    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        start_idx = e * length_per_electrode
        end_idx = (e + 1) * length_per_electrode

        tar_e = signal_tar[:, start_idx:end_idx]
        ntar_e = signal_ntar[:, start_idx:end_idx]
        all_e = input_signal[:, start_idx:end_idx]

        signal_tar_cov[e] = np.cov(tar_e, rowvar=False)
        signal_ntar_cov[e] = np.cov(ntar_e, rowvar=False)
        signal_all_cov[e] = np.cov(all_e, rowvar=False)

    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]


def plot_trunc_mean(eeg_tar_mean, eeg_ntar_mean, subject_name, time_index,
                    E_val, electrode_name_ls, y_limit=np.array([-5, 8]), fig_size=(12, 12)):
    """
    Plot and save target and non-target sample means in a 4x4 grid.
    """
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.flatten()

    for e in range(E_val):
        ax = axes[e]
        ax.plot(time_index, eeg_tar_mean[e, :], color='r', label='Target')
        ax.plot(time_index, eeg_ntar_mean[e, :], color='b', label='Non-Target')
        ax.set_title(electrode_name_ls[e], fontsize=10)
        ax.set_ylim(y_limit)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")

    fig.suptitle(f"{subject_name} Target vs. Non-Target Sample Means", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure to subject folder
    out_dir = os.path.join(os.getcwd(), subject_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plt.savefig(os.path.join(out_dir, "Mean.png"))
    plt.close()


def plot_trunc_cov(eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14, 12)):
    """
    Plot and save electrode-specific covariance matrices in a 4x4 grid.
    """
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.flatten()

    for e in range(E_val):
        ax = axes[e]
        X, Y = np.meshgrid(time_index, time_index)
        c = ax.contourf(X, Y, eeg_cov[e, :, :], cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(electrode_name_ls[e], fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Time (ms)")

    fig.suptitle(f"{subject_name} {cov_type} Covariance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.getcwd(), subject_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plt.savefig(os.path.join(out_dir, f"Covariance_{cov_type}.png"))
    plt.close()