from numpy.lib.npyio import save
from scipy.io import loadmat, savemat
from scipy import signal
import wfdb.processing
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
from final_preprocessing_pipeline import high_pass_filter, morlet_wavelet
from sklearn.utils import shuffle

# start count from 29136 for Ningbo
# get all filenames that match the data we want
# loop through all headers, open, check headers
header_num = 10
# header_num = 30
datasets = {
    # "WFDB_CPSC2018": ["A", 6877, 4],
    # "WFDB_CPSC2018_2": ["Q", 3581, 4],
    # "WFDB_StPetersburg": ["I", 75, 4],
    # "WFDB_PTB": ["S", 549, 4],
    "WFDB_PTBXL": ["HR", 21837, 5],
    "WFDB_ChapmanShaoxing": ["JS", 8900, 5],
    "WFDB_Ningbo": ["JS", 45551, 5],
}
data_path = "/om2/user/sadhana/data/"
ts_savepath = "/om2/user/sadhana/time-series-data/"
tf_savepath = "/om2/user/sadhana/time-frequency-data/"
relevant_labels_true = [429622005, 164931005, 59931005, 164934002, 426783006]
window_length = 10
RESAMPLE_FREQUENCY = 500
NUM_LEADS = 12
count = 2860
print("STARTED RUNNING")
for dataset, info in datasets.items():
    relevant_data = []
    labels = []
    from_dataset = []
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    path = data_path + dataset + "/"
    header_num = info[1]
    chara = info[0]
    for data_piece in range(1, header_num + 1):
        numZeros = info[2]
        try:
            f = open(
                path
                + chara
                + "0" * (numZeros - len(str(data_piece)))
                + str(data_piece)
                + ".hea",
                "r",
            )
            f = f.readlines()
            f = [i.strip() for i in f]
            dx = f[15][4:].split(",")
            dx = [int(i.strip()) for i in dx]
            relevant_diagnoses = [x for x in dx if x in relevant_labels_true]
            if len(relevant_diagnoses) > 0:
                print("dataset: ", dataset)
                print("processing data # :", data_piece, "out of", header_num)
                print("transformed into data # :", count)
                # load data
                train_or_val = np.random.choice(2, p=[0.7, 0.3])
                record = wfdb.rdrecord(
                    path
                    + chara
                    + "0" * (numZeros - len(str(data_piece)))
                    + str(data_piece)
                )
                signals, fields = wfdb.rdsamp(
                    path
                    + chara
                    + "0" * (numZeros - len(str(data_piece)))
                    + str(data_piece)
                )

                # normalize and resample data
                signals = wfdb.processing.normalize_bound(signals)
                signals = scipy.signal.resample(
                    signals, int(fields["fs"] * signals.shape[0] / RESAMPLE_FREQUENCY),
                )

                # cropping loop
                while len(signals) >= window_length * RESAMPLE_FREQUENCY:
                    # generate time frequency data and save
                    tf_data = []
                    savesignals = []
                    for lead in range(NUM_LEADS):
                        to_transform = signals[
                            : window_length * RESAMPLE_FREQUENCY, lead
                        ]
                        savesignals.append(
                            high_pass_filter(to_transform, RESAMPLE_FREQUENCY,)
                        )
                        tf_data.append(
                            morlet_wavelet(to_transform, RESAMPLE_FREQUENCY,)
                        )
                    tf_data = np.asarray(tf_data)
                    savesignals = np.asarray(savesignals)
                    print("shape of data: ", tf_data.shape, savesignals.shape)
                    np.save(
                        ts_savepath + str(count), savesignals,
                    )
                    np.save(tf_savepath + str(count), tf_data)
                    label = 1
                    if 426783006 in relevant_diagnoses:
                        label = 0
                    # generate labels
                    if train_or_val == 0:  # append to train set
                        train_data.append(str(count) + ".npy")
                        train_labels.append(label)
                    else:
                        val_data.append(str(count) + ".npy")
                        val_labels.append(label)
                    count += 1
                    signals = signals[window_length * RESAMPLE_FREQUENCY :, :]
        except Exception as e:
            print(e)
        if data_piece % 100 == 0:
            train_df_so_far = pd.DataFrame(
                {
                    "filename": train_data,
                    "y": train_labels,
                    "dataset": [dataset for i in range(len(train_data))],
                }
            )
            train_df_so_far.to_csv(dataset + "_train" + str(data_piece), index=False)
            print(len(val_labels), len(val_data))
            val_df_so_far = pd.DataFrame(
                {
                    "filename": val_data,
                    "y": val_labels,
                    "dataset": [dataset for i in range(len(val_data))],
                }
            )
            val_df_so_far.to_csv(dataset + "_val" + str(data_piece), index=False)
    # relevant_data = np.array(relevant_data)
    # label = np.ones(len(relevant_data,))
    # relevant_data, labels, from_dataset = shuffle(relevant_data, labels, from_dataset)
    train_df_so_far = pd.DataFrame(
        {
            "filename": train_data,
            "y": train_labels,
            "dataset": [dataset for i in range(len(train_data))],
        }
    )
    train_df_so_far.to_csv(dataset + "_train_total.csv", index=False)

    val_df_so_far = pd.DataFrame(
        {
            "filename": val_data,
            "y": val_labels,
            "dataset": [dataset for i in range(len(val_data))],
        }
    )
    val_df_so_far.to_csv(dataset + "_val_total.csv", index=False)
    print(dataset, "saved!")
    print("positive samples: ", sum(labels))
    print("negative samples", len(labels) - sum(labels))
    # print(df)

