import collections
import os
from pathlib import Path

import numpy as np
import pandas as pd
# from mne_bids import get_entities_from_fname
# from natsort import natsorted

from sample_code.read_datasheet import read_clinical_excel
from sample_code.study import _sequential_aggregation
from sample_code.utils import _subsample_matrices_in_time, _smooth_matrix


def read_participants_tsv(root, participant_id=None, column_keys=None):
    """Read in ``participants.tsv`` file.

    Allows one to query the columns of the file as
    a DataFrame and say obtain the list of Engel scores.
    """
    participants_tsv_fname = os.path.join(root, 'participants.tsv')

    participants_df = pd.read_csv(participants_tsv_fname, delimiter='\t')
    if participant_id is not None:
        participants_df = participants_df[participants_df['participant_id'] == participant_id]

    if column_keys is not None:
        if not isinstance(column_keys, list):
            column_keys = [column_keys]
        participants_df = participants_df[column_keys]
    return participants_df


def load_ictal_frag_data(deriv_path, excel_fpath, patient_aggregation_method=None, expname='sliced',
                         reference='average'):
    ignore_pats = [
        "jh107",
    ]

    modelname = "fragility"
    kind = 'ieeg'

    # load in data as patient dictionary of lists
    datadir = Path(deriv_path) / f"{expname}/{modelname}"  # / reference / 'ieeg'
    # load in sliced datasets
    patient_result_dict = _load_patient_dict(datadir, kind=kind, verbose=True)

    unformatted_X = []
    y = []
    sozinds_list = []
    onsetwin_list = []
    subjects = []

    for subject, datasets in patient_result_dict.items():
        #     print(subject, len(datasets))
        # read in Excel database
        pat_dict = read_clinical_excel(excel_fpath, subject=subject)
        soz_chs = pat_dict["SOZ_CONTACTS"]
        #     soz_chs = pat_dict['RESECTED_CONTACTS']
        outcome = pat_dict["OUTCOME"]

        # extract list of the data from subject
        mat_list = [dataset["mat"] for dataset in datasets]
        mat_list = _subsample_matrices_in_time(mat_list)
        ch_names_list = [dataset["ch_names"] for dataset in datasets]

        # get a nested list of all the SOZ channel indices
        _sozinds_list = [
                            [ind for ind, ch in enumerate(ch_names_list[0]) if ch in soz_chs]
                        ] * len(datasets)

        # get a list of all the onset indices in the heatmaps
        _onsetwin_list = [dataset["onsetwin"] for dataset in datasets]
        if subject in ignore_pats:
            continue
        if not _sozinds_list:
            print(subject)
        if all(sozlist == [] for sozlist in _sozinds_list):
            print("Need to skip this subject: ", subject)
            continue

        if outcome == "NR":
            continue

        # aggregate and get (X,Y) pairs
        if patient_aggregation_method == "median":
            if len(np.unique(_onsetwin_list)) > 1:
                print(subject, _onsetwin_list)
                raise RuntimeError("Has more then one onset times... can't aggregate.")
            if not all(set(inds) == set(_sozinds_list[0]) for inds in _sozinds_list):
                print(_sozinds_list)
                raise RuntimeError(
                    "Each dataset has different soz inds... can't aggregate."
                )
            if not all(mat.shape[0] == mat_list[0].shape[0] for mat in mat_list):
                raise RuntimeError("Can't aggregate...")

            for mat in mat_list:
                mat = _smooth_matrix(mat, window_len=8)

            mat = np.max(mat_list, axis=0)
            y.append(outcome)
            unformatted_X.append(mat)
            subjects.append(subject)
            sozinds_list.append(_sozinds_list[0])
            onsetwin_list.append(_onsetwin_list[0])
        elif patient_aggregation_method is None:
            _X, _y, _sozinds_list = _sequential_aggregation(
                mat_list, ch_names_list, _sozinds_list, outcome
            )
            if len(_y) != len(_sozinds_list):
                print(subject, len(_y), len(_sozinds_list))
                continue
            unformatted_X.extend(_X)
            y.extend(_y)
            sozinds_list.extend(_sozinds_list)
            onsetwin_list.extend(_onsetwin_list)
            subjects.extend([subject] * len(_y))

    print(
        len(unformatted_X), len(y), len(sozinds_list), len(subjects), len(onsetwin_list)
    )
    return unformatted_X, y, subjects, sozinds_list, onsetwin_list


def _load_patient_dict(datadir, kind="ieeg", expname='sliced', verbose=True):
    """Load from datadir, sliced datasets as a dictionary <subject>: <list of datasets>."""
    pats_to_avg = [
        "umf002",
        "umf004",
        "jh103",
        "ummc005",
        "ummc007",
        "ummc008",
        "ummc009",
        "pt8",
        "pt10",
        "pt11",
        "pt12",
        "pt16",
        "pt17",
        "la00",
        "la01",
        "la02",
        "la03",
        "la04",
        "la05",
        "la06",
        "la07",
        "la08",
        "la10",
        "la11",
        "la12",
        "la13",
        "la15",
        "la16",
        "la20",
        "la21",
        "la22",
        "la23",
        "la24",
        "la27",
        "la28",
        "la29",
        "la31",
        "nl01",
        "nl02",
        "nl03",
        "nl04",
        "nl05",
        "nl06",
        "nl07",
        "nl08",
        "nl09",
        "nl13",
        "nl14",
        "nl15",
        "nl16",
        "nl18",
        "nl21",
        "nl23",
        "nl24",
        "tvb1",
        "tvb2",
        "tvb5",
        "tvb7",
        "tvb8",
        "tvb11",
        "tvb12",
        "tvb14",
        "tvb17",
        "tvb18",
        "tvb19",
        "tvb23",
        "tvb27",
        "tvb28",
        "tvb29",
    ]

    patient_result_dict = collections.defaultdict(list)
    num_datasets = 0

    # get all files inside experiment
    trimmed_npz_fpaths = [x for x in datadir.rglob("*npz")]

    # get a hashmap of all subjects
    subjects_map = {}
    for fpath in trimmed_npz_fpaths:
        params = get_entities_from_fname(
            os.path.basename(fpath).split(f"{expname}-")[1]
        )
        subjects_map[params["subject"]] = 1

    if verbose:
        print(f'Got {len(subjects_map)} subjects')

    # loop through each subject
    subject_list = natsorted(subjects_map.keys())
    for subject in subject_list:
        if subject in pats_to_avg:
            #             print("USING AVERAGE for: ", fpath)
            reference = "average"
        else:
            reference = "monopolar"
        subjdir = Path(datadir / reference / kind)
        fpaths = [x for x in subjdir.glob(f"*sub-{subject}_*npz")]

        # load in each subject's data
        for fpath in fpaths:
            # load in the data and append to the patient dictionary data struct
            with np.load(fpath, allow_pickle=True) as data_dict:
                data_dict = data_dict["data_dict"].item()
                patient_result_dict[subject].append(data_dict)

            num_datasets += 1

    if verbose:
        print("Got ", num_datasets, " datasets.")
        print("Got ", len(patient_result_dict), " patients")
        print(patient_result_dict.keys())

    return patient_result_dict
