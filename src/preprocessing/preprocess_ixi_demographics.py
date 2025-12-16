import argparse
import itertools
import os

import numpy as np
import pandas as pd
from loguru import logger


def check_fix_unique_id(
    clinical_df: pd.DataFrame,
    identifier: str,
    columns_consistency_list: list,
    report_filename: str,
) -> pd.DataFrame:
    """quality check function: checks unique id, and keeps subject with same values over
    the list of columns defined in columns_consistency_list

    Args:
        clinical_df (pd.DataFrame): data
        identifier (str): column name for subject identification
        columns_consistency_list (list): list of columns name used to check if
            the value is the same, if more rows refer to same subject id,
            but the values are the same, then the common
            values of this subject are kept
        report_filename (str): text file with the summary of all changes made in the data
            for visual inspection

    Returns:
        pd.DataFrame: corrected data
    """
    # check if there are unique ID
    clinical_df_id_gr = clinical_df.groupby(identifier)
    not_unique_id = clinical_df_id_gr[identifier].value_counts() > 1
    # get all groups with more than 1 rows:
    all_unique_id = list(clinical_df_id_gr.groups.keys())
    bad_id = list(itertools.compress(all_unique_id, not_unique_id))
    good_id = list(itertools.compress(all_unique_id, ~not_unique_id))
    good_subgroup_df = clinical_df[clinical_df[identifier].isin(good_id)].copy(deep=True)

    if len(bad_id) == 0:
        return clinical_df

    logger.warning(f"Number of subject with more than 1 raw in the clinical data are: {len(bad_id)}")
    # fix double id with report saved in data folder:
    corrected_subj_data, removed_subj = [], []
    with open(report_filename, "w", encoding="utf8") as f:
        # visual check of identified bad id
        for i_bad_id in bad_id:
            this_subj = clinical_df_id_gr.get_group(i_bad_id)
            same_values_over_rows = [(this_subj[this_condition] == this_subj[this_condition].iloc[0]).all() for this_condition in columns_consistency_list]
            temp = {}
            if all(same_values_over_rows):
                # keep only concordant values:
                for i_column in this_subj.columns:
                    first_val = this_subj[i_column].iloc[0]
                    if (this_subj[i_column] == first_val).all():
                        temp[i_column] = first_val
                    else:
                        temp[i_column] = np.nan
                corrected_subj_data.append(temp)
                print(f"Updated subj: {i_bad_id}", file=f)
                print("Original data:", file=f)
                print(this_subj, file=f)
                print("---- corrected into---- :", file=f)
                print(pd.DataFrame([temp]), file=f)
            else:
                removed_subj.append(i_bad_id)
            print(
                "------------------------------------------------------------------",
                file=f,
            )

        print(
            f"removed subject because discordant conditional feature values: {removed_subj}",
            file=f,
        )

    # combine corrected data with the good once:
    corrected_df = pd.DataFrame.from_dict(corrected_subj_data)

    full_corrected_df = pd.concat([corrected_df, good_subgroup_df], ignore_index=True)
    full_corrected_df = full_corrected_df.sort_values([identifier], ascending=(True)).reset_index(drop=True)

    return full_corrected_df


def main_demographic_processing(
    clinical_file: str,
    clinical_file_post_processing: str,
    identifier: str = "IXI_ID",
    bins_categorical_columns: dict = None,
    columns_consistency_list: list = None,
):
    """functions that reads the clinical or demographic data and defines groups based
        bins_categorical_columns

    NOTE: this function does some checks and writes a report of changed, check if
    new data require more checks or more categories definitions

    Args:
        clinical_file (str): input file
        clinical_file_post_processing (str): file of the postprocessed data
        identifier (str): column name used to identify subject, used to check
            unique subject id in the data
        bins_categorical_columns (dict, optional): key:value where key=column name
            of the data, value=number of categories
            used to define categories based on data distribution. Defaults to {'AGE':3}
            here we use the column AGE and define 3 groups.
        columns_consistency_list (list, optional): list of column names used on the quality check.
            Defaults to ['gender_m0_f1', 'AGE_categ_int'].
    """

    if bins_categorical_columns is None:
        bins_categorical_columns = {"AGE": 3}
    if columns_consistency_list is None:
        columns_consistency_list = ["gender_m0_f1", "AGE_categ_int"]

    assert os.path.isfile(clinical_file), f"input file does not exist: {clinical_file}"

    if clinical_file.endswith(".csv"):
        clinical_df = pd.read_csv(clinical_file, index_col=False)
    elif clinical_file.endswith(".xls") or clinical_file.endswith(".xlsx"):
        clinical_df = pd.read_excel(clinical_file, index_col=False)
    else:
        raise ValueError(f"Unknown input file format, please use .csv or .xls[x]: {clinical_file}")

    clinical_df[identifier] = "IXI" + clinical_df[identifier].astype(str).apply(
        "{0:0>3}".format  # pylint: disable=consider-using-f-string
    )

    # define age categories based on data distribution:
    for key, val in bins_categorical_columns.items():
        logger.info(f"converting {key} into {val} bins based on the distribution into '{key + '_categ'}' column")
        clinical_df[key + "_categ"] = pd.qcut(clinical_df[key], q=val)
        clinical_df[key + "_categ_int"] = clinical_df[key + "_categ"].cat.codes
        clinical_df.loc[clinical_df[key].isna(), key + "_categ_int"] = clinical_df[key][clinical_df[key].isna()].copy(deep=True)

    if "SEX_ID (1=m, 2=f)" in clinical_df.columns:
        clinical_df["gender_m0_f1"] = clinical_df["SEX_ID (1=m, 2=f)"] - 1

    # flatten duplicated subject rows into one row
    report_filename = clinical_file_post_processing + ".qc-report.txt"
    full_corrected_df = check_fix_unique_id(clinical_df, identifier, columns_consistency_list, report_filename)

    # save
    full_corrected_df.to_csv(clinical_file_post_processing, index=False)
    logger.info(f"processed file saved (see the report for more details): {clinical_file_post_processing}")


def main():
    parser = argparse.ArgumentParser("Preprocessing script for the IXI demographic data. Flattens duplicated subject rows into one row and creates age bins.")
    parser.add_argument("input_file", nargs=1, help="Path to the input file to preprocess.")
    parser.add_argument("output_file", nargs=1, help="Path to output location.")
    args = parser.parse_args()

    main_demographic_processing(args.input_file[0], args.output_file[0])


if __name__ == "__main__":
    main()
