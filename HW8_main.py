import os
import numpy as np
import scipy.io as sio
from HW8Fun import produce_trunc_mean_cov, plot_trunc_mean, plot_trunc_cov

bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                     'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
time_index = np.linspace(0, 800, 25)
subject_name = 'K114'

parent_dir = '/Users/julia/Documents/GitHub/BIOS-584-test'
data_path = os.path.join(parent_dir, 'data', 'K114_001_BCI_TRN_Truncated_Data_0.5_6.mat')

subject_dir = os.path.join(parent_dir, subject_name)
if not os.path.exists(subject_dir):
    os.mkdir(subject_dir)

eeg_trunc_obj = sio.loadmat(data_path)
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = np.squeeze(eeg_trunc_obj['Type'], axis=1)

results = produce_trunc_mean_cov(eeg_trunc_signal, eeg_trunc_type, electrode_num)

plot_trunc_mean(results[0], results[1], subject_name, time_index, electrode_num, electrode_name_ls)
plot_trunc_cov(results[2], "Target", time_index, subject_name, electrode_num, electrode_name_ls)
plot_trunc_cov(results[3], "Non-Target", time_index, subject_name, electrode_num, electrode_name_ls)
plot_trunc_cov(results[4], "All", time_index, subject_name, electrode_num, electrode_name_ls)