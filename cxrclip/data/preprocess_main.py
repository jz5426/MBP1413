import os
import pandas as pd
import numpy as np
import ast
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV_PARENT_DIR = '/cluster/projects/mcintoshgroup/CXR-CLIP/preprossed_dataset_csv/'


def rename_normal_labels(csv_file_path, display_label_column='class'):
    """
    Convert the label 'Normal' to 'No [other_class]' in-place.

    Assumptions:
    - The CSV contains exactly two unique values in `display_label_column`
    - One of them is 'Normal'
    - The other is the disease class
    """
    
    # load csv
    df = pd.read_csv(csv_file_path)

    if display_label_column not in df.columns:
        raise ValueError(f"Column '{display_label_column}' not found in CSV")

    unique_classes = df[display_label_column].dropna().unique()
    
    # sanity check
    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected exactly 2 unique classes, found {len(unique_classes)}: {unique_classes}"
        )
    if "Normal" not in unique_classes:
        raise ValueError(f"'Normal' not found in classes: {unique_classes}")

    # identify disease class
    disease_class = [c for c in unique_classes if c != "Normal"][0]
    replacement_label = f"No {disease_class.lower().strip()}"

    # replace Normal
    df.loc[df[display_label_column] == "Normal", display_label_column] = replacement_label

    # save back to same path
    df.to_csv(csv_file_path, index=False)
    print(f'saved to {csv_file_path}')

    return

def combine_csvs(csvs=[], destination=''):
    dfs = [pd.read_csv(csv) for csv in csvs]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(destination, index=False)
    print(f'saved to {destination}')

if __name__ == '__main__':
    # chexchonet_slvh
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/slvh_test.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/slvh_val.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/slvh_train.csv'))

    # chexchonet_dlv
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/dlv_test.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/dlv_val.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/dlv_train.csv'))

    # chexchonet_composite_slvh_dlv
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/composite_slvh_dlv_test.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/composite_slvh_dlv_val.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'chexchonet224/composite_slvh_dlv_train.csv'))

    # rsna_pneumonia
    # main(os.path.join(CSV_PARENT_DIR, 'rsna/rsna_test.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'rsna/rsna_valid.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'rsna/rsna_train.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'rsna/rsna_all.csv'))

    # siim_pneumothorax
    # main(os.path.join(CSV_PARENT_DIR, 'siim/siim_test.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'siim/siim_valid.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'siim/siim_train.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'siim/siim_all.csv'))

    # shenzhenxray #TODO: no valid and train data for this one
    # main(os.path.join(CSV_PARENT_DIR, 'shenzhenXray/shenzhen_test_eval.csv'))

    # montgomery #TODO: no valid and train data for this one
    # main(os.path.join(CSV_PARENT_DIR, 'montgomery/montgomery_test_eval.csv'))

    # covidkaggle
    # rename_normal_labels(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/test_binaryCovid.csv'))
    # rename_normal_labels(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/valid_binaryCovid.csv'))
    # rename_normal_labels(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/train_binaryCovid.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/test_4classes.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/valid_4classes.csv'))
    # main(os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/train_4classes.csv'))

    # combine csv
    combine_csvs(
        [os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/test_binaryCovid.csv'), os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/valid_binaryCovid.csv')],
        os.path.join(CSV_PARENT_DIR, 'covid_kaggle_256/testAndValid_binaryCovid.csv')
    )
    pass