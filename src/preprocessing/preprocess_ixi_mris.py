import argparse
import glob
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchio as tio
from HD_BET.run import run_hd_bet
from loguru import logger
from tqdm import tqdm


def insert_str_to_file(file: str, add_str: str):
    """define new filename by adding a string at the end

    Args:
        file (str): full file
        add_str (str): string to add at the end

    Returns:
        str: new string with new ending
            new_file_str = os.path.join(file_dir, filename + add_str + ext)
    """

    file_dir, filename = os.path.split(file)
    temp = filename.split(".")
    filename = temp[0]
    ext = "." + ".".join(temp[1:])
    new_file_str = os.path.join(file_dir, filename + add_str + ext)

    return new_file_str


def reorient(
    this_subj_data: tio.data.subject,
    this_subject_id: str,
    this_subj_out_dir: str,
    modalities: list,
) -> dict:
    """mreorient to canonincal usig torch io functions

    Args:
        this_subj_data (_type_): _description_
        this_subject_id (str): _description_
        this_subj_out_dir (str): _description_
        modalities (list): _description_

    Returns:
        dict: _description_
    """

    this_file_reorient = {}
    for i_mod in modalities:
        this_file_reorient[i_mod] = os.path.join(
            this_subj_out_dir,
            this_subject_id + "-" + i_mod + "-reorient_resample.nii.gz",
        )
    # check if already computed
    reorient_done = [os.path.isfile(this) for this in this_file_reorient.values()]
    if all(reorient_done):
        logger.info(f"Step reorient and resample (subj {this_subject_id}): already done")
        return this_file_reorient

    logger.info(f"reorient_rescale (subj:{this_subject_id})")
    transforms = [
        tio.ToCanonical(),  # to RAS
        tio.Resample((1, 1, 1)),  # to 1 mm iso
    ]
    transform = tio.Compose(transforms)
    this_subj_data = transform(this_subj_data)

    for i_mod in modalities:
        this_file = os.path.join(
            this_subj_out_dir,
            this_subject_id + "-" + i_mod + "-reorient_resample.nii.gz",
        )
        this_subj_data[i_mod].save(this_file)

    logger.info(f"Step reorient and resample (subj {this_subject_id}): done")
    return this_file_reorient


def correct_mri_bias_ants(filename: str):
    """
    Corrects the MRI image from `filename` using Ants
    """
    assert Path(filename).exists()

    output_file = insert_str_to_file(filename, "_n4_bias_ants")
    if Path(output_file).exists():
        logger.info(f"Bias correction already complete for {filename} -> {output_file}")
        return output_file

    command = (
        os.path.join(os.environ["ants_dir"], "N4BiasFieldCorrection") + " -d 3 -i " + filename + " -o " + output_file + " -c [100x100x100x100,0.0000000001]"
    )
    logger.info(f"running '{command}'")
    subprocess.check_call(command, shell=True)

    return output_file


def main_bias_correction(data_dict: dict, this_subj_out_dir: str, this_subject_id: str):
    bias_corrected_data_dict = {}

    for i_mod, this_file in data_dict.items():
        # define output file
        this_out_file = os.path.join(
            this_subj_out_dir,
            this_subject_id + "-" + i_mod + "_n4_bias_ants.nii.gz",
        )

        # check if already computed
        if os.path.isfile(this_out_file):
            logger.info(f"Step: bias correction (subj:{this_subject_id}-mod{i_mod}): already done")
            bias_corrected_data_dict[i_mod] = this_out_file
        else:
            logger.info(f"Step: bias correction (subj:{this_subject_id}-mod-{i_mod}): running ...")
            if os.path.isfile(this_file):
                bias_corrected = correct_mri_bias_ants(this_file)
                bias_corrected_data_dict[i_mod] = bias_corrected
            else:
                logger.warning(f"input file for bias correction does not exists: {this_file}")

    return bias_corrected_data_dict


def main_brain_extraction(data_dict: dict, brain_extraction_mode: str = "accurate"):
    # computes brain extraction
    computed_brain_extr_dict = {this_key: insert_str_to_file(file=this_value, add_str="_brain_extracted") for (this_key, this_value) in data_dict.items()}

    out_files = list(computed_brain_extr_dict.values())
    if all([os.path.isfile(this) for this in out_files]):
        logger.info(f"Step: brain_extraction: already done for {out_files}.")
        return computed_brain_extr_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = 0 if device == "cuda" else "cpu"

    input_files = list(data_dict.values())
    logger.info(f"Step: brain_extraction: running for {out_files}...")
    run_hd_bet(
        input_files,
        out_files,
        mode=brain_extraction_mode,
        keep_mask=True,
        device=gpu_id,
        bet=True,
    )

    return computed_brain_extr_dict


def main_intensity_norm_crop(data_dict: dict, target_shape: tuple = (256, 256, 256)):
    # intensity rescaling between 0-1 and crop/pad to 256^3
    norm_cropped_dict = {this_key: insert_str_to_file(file=this_value, add_str="_norm_cropped") for (this_key, this_value) in data_dict.items()}
    out_list = list(norm_cropped_dict.values())
    if all([os.path.isfile(this) for this in out_list]):
        logger.info(f"Step: intensity norm and crop: already done for {out_list}.")
        return norm_cropped_dict

    # define transformation
    transforms_final = [
        tio.RescaleIntensity((0, 1)),
        tio.CropOrPad(target_shape),
    ]
    transform = tio.Compose(transforms_final)

    for i_mod, this_file in data_dict.items():
        logger.info(f"Step: normalize and crop: running for {out_list}...")
        temp = tio.ScalarImage(this_file)
        temp_transformed = transform(temp)
        # save as nifty
        temp_transformed.save(norm_cropped_dict[i_mod])

    return norm_cropped_dict


def warp_to_mni_space_core(input_data, main_out_path, templates_mni):
    # check if templates exists
    templ_exists = [os.path.isfile(this) for this in templates_mni]
    assert all(templ_exists), f"template files do not exist:\n{templates_mni}"
    ref_mni = templates_mni[0]
    ref_mni_mask = templates_mni[1]

    this_subj_path, _ = os.path.split(input_data)

    os.makedirs(main_out_path, exist_ok=True)

    # warp to MNI:
    # define inputs:
    brain_extracted = input_data
    img_mask = insert_str_to_file(brain_extracted, "_mask")
    assert os.path.isfile(img_mask), f"img_mask does not exists: {img_mask}"

    # define outputs filenames
    brain_extracted_mni = insert_str_to_file(brain_extracted, "_mni")
    if not os.path.isfile(brain_extracted_mni):
        logger.info(f"running: {os.path.basename(this_subj_path)} antsRegistration to MNI")
        command = (
            os.path.join(os.environ["ants_dir"], "antsRegistration")
            + " -d 3 -o ["
            + this_subj_path
            + "/mni_transform"
            + ","
            + brain_extracted_mni
            + "]"
            + " -x ["
            + ref_mni_mask
            + ","
            + img_mask
            + "]"
            + " -n BSpline -u 0 -r ["
            + ref_mni
            + ","
            + brain_extracted
            + ",1]"
            + " -t Rigid[0.1] -m MI["
            + ref_mni
            + ","
            + brain_extracted
            + ",1,32,Regular,0.25] -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox"
            + " -t Affine[0.1] -m MI["
            + ref_mni
            + ","
            + brain_extracted
            + ",1,32,Regular,0.25] -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox"
            + " -t SyN[0.1,3,0] -m CC["
            + ref_mni
            + ","
            + brain_extracted
            + ",1,2] -c [200x200x200x50,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox",
        )

        subprocess.check_call(command, shell=True)

        # check if ants_registration worked
        if os.path.isfile(brain_extracted_mni):
            logger.info("antsRegistration to MNI: done\n")

    post_process(brain_extracted_mni, main_out_path)


def post_process(this_input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    this_dir, this_filename = os.path.split(this_input_file)
    filename = this_filename.split(".")[0]
    ext = "." + ".".join(this_filename.split(".")[1:])

    # output file in nifti and pickle
    out_file = os.path.join(this_dir, filename + "_norm_reshape" + ext)
    pickle_filename = filename.replace("-", "_")  # necessary for matching with clinical data
    out_file_pickle = os.path.join(output_dir, pickle_filename + "_norm_reshape.pickle")

    if os.path.isfile(out_file_pickle):
        logger.info(f"already computed output: {out_file_pickle}")
        return

    target_shape = (256, 256, 256)
    this_subj_data = tio.ScalarImage(this_input_file)
    # final transformation with intensity rescaling between 0-1 and crop/pad to 256^3
    transforms_final = [
        tio.Resize(target_shape=target_shape),
        tio.RescaleIntensity((0, 1)),
    ]
    transform = tio.Compose(transforms_final)
    this_subj_data = transform(this_subj_data)
    # save as nifty
    this_subj_data.save(out_file)
    assert os.path.isfile(out_file)

    this_img = this_subj_data.numpy().squeeze()
    sample = {"image": this_img}
    with open(out_file_pickle, "wb") as f:
        pickle.dump(sample, f)  # save as pickle:
    logger.info(f"computed output: {out_file_pickle}")


def _preprocess_mri(args: Tuple[Dict, str, str, List, Tuple[int, int, int], str]):
    """calls all preprocessing steps for MRI data

    Args:
        args: Tuple of args passed through multiprocessing
    """
    (
        this_subj_data,
        interim_path,
        output_path,
        modalities,
        target_shape,
        mni_template_path,
    ) = args
    this_subject_id = this_subj_data.subject_id
    try:
        brain_extraction_mode = "accurate"

        this_subj_out_dir = os.path.join(interim_path, this_subject_id)
        os.makedirs(this_subj_out_dir, exist_ok=True)

        # reorient
        reoriented_data_dict = reorient(this_subj_data, this_subject_id, this_subj_out_dir, modalities)

        # bias correction
        bias_corrected_data = main_bias_correction(
            data_dict=reoriented_data_dict,
            this_subj_out_dir=this_subj_out_dir,
            this_subject_id=this_subject_id,
        )

        # brain_extraction: (this step faster in GPU)
        computed_brain_extr_dict = main_brain_extraction(data_dict=bias_corrected_data, brain_extraction_mode=brain_extraction_mode)

        # intensity range 0-1 and crop/pad to target_shape for each  modality
        norm_cropped_dict = main_intensity_norm_crop(data_dict=computed_brain_extr_dict, target_shape=target_shape)

        # save as pickle
        logger.info(f"Step: save as pickle (subj:{this_subject_id})")
        for i_mod, this_file in norm_cropped_dict.items():
            pickle_filename = this_subject_id.replace("-", "_") + "_" + i_mod + "_processed.pickle"  # necessary for matching with clinical data

            brain_extracted_pickle_file = os.path.join(output_path, "subj_space", i_mod, pickle_filename)
            os.makedirs(os.path.dirname(brain_extracted_pickle_file), exist_ok=True)

            temp = tio.ScalarImage(this_file)
            this_img = temp.numpy().squeeze()
            with open(brain_extracted_pickle_file, "wb") as f:
                pickle.dump({"image": this_img}, f)

        if mni_template_path:
            mni_template_str = "mni_icbm152_t1_tal_nlin_sym_09c.nii"
            brain_mask_mni = "mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"
            templates_mni = [
                os.path.join(mni_template_path, mni_template_str),
                os.path.join(mni_template_path, brain_mask_mni),
            ]  # first the ref_mni then the mask!
            # check if templates exists
            template_exists = all(os.path.isfile(this) for this in templates_mni)
            assert template_exists, (
                "Missing templates, please download the 'ICBM 2009c Nonlinear Symmetric' "
                "template from https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)"
            )

            # get processed input data
            # get all T1 brain extracted
            t1_map_list = list(
                glob.glob(
                    this_subj_out_dir + "/**/*-T1-*brain_extracted.nii.gz",
                    recursive=True,
                )
            )

            os.makedirs(os.path.join(output_path, "mni_space"), exist_ok=True)
            for this_subj_input in t1_map_list:
                logger.info(f"processing: \n{this_subj_input}\n")
                warp_to_mni_space_core(
                    this_subj_input,
                    os.path.join(output_path, "mni_space", "T1"),
                    templates_mni,
                )

        return (True, None)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return (False, f"Failed to process {this_subject_id}: {e}")


def preprocessing_mri_pipeline(
    raw_data_path: str,
    intermediate_path: str,
    mni_template_path: Optional[str],
    output_path: str,
    modalities: List[str],
    download_data: bool = False,
    target_shape: Tuple[int, int, int] = (256, 256, 256),
    batch_bounds: Tuple[int, int] = (0, -1),
    cores: int = -1,
):
    """preprocessing of each MRI including:
    reorientation, bias correction, multi-modal registration and intensity normalization

    Args:
        raw_data_path: where to download the raw data
        intermediate_path: location for intermediate processed files
        mni_template_path: optional location of the MNI templates for warping to MNI space
        output_path: location of output files
        modalities: List of data modalities to process (e.g. ["T1", "T2", "PD"])
        download_data: set to True to download via TorchIO, otherwise assume dataset has already been downloaded
        target_shape: output 3D image shape as a 3-tuple
        batch_bounds: list indices specifying a batch
        cores: number of cores to use with multiprocessing. None uses all available, 1 disables use of multiprocessing
    """

    # download public dataset:
    ixi_dataset = tio.datasets.IXI(
        root=raw_data_path,
        modalities=modalities,
        download=download_data,
    )

    logger.info(f"Number of subjects in dataset: {len(ixi_dataset)}")

    # preprocees
    interim_path = os.path.join(intermediate_path, "interim")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running in {device} mode")

    batch_range = list(range(len(ixi_dataset)))[batch_bounds[0] : batch_bounds[1]]
    logger.info(f"Number of subjects in batch: {len(batch_range)}")
    process_list = [
        (
            ixi_dataset[i_subj],
            interim_path,
            output_path,
            modalities,
            target_shape,
            mni_template_path,
        )
        for i_subj in batch_range
    ]

    # prepocess each subject (data in subject space)
    if len(process_list) > 1 and (cores is None or cores > 1):
        pbar = tqdm(total=len(process_list))
        with mp.Pool(cores) as p:
            for i, result in enumerate(p.imap_unordered(_preprocess_mri, process_list)):
                pbar.update(i)
                if not result[0]:
                    logger.error(f"Error processing subject: {result[1]}")
    else:
        for i, result in tqdm(enumerate(map(_preprocess_mri, process_list))):
            if not result[0]:
                logger.error(f"Error processing subject: {result[1]}")


def main():
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser("Preprocess MRI data")
    parser.add_argument(
        "output_dir",
        nargs=1,
        help="The output directory for processed images.",
    )
    parser.add_argument(
        "-m",
        "--modalities",
        help="List of modalities, comma separated (e.g. T1,T2)",
        default="T1",
    )
    parser.add_argument(
        "-i",
        "--intermediate-data",
        help="Directory for intermediate data",
        default=None,
    )
    parser.add_argument(
        "--raw-data-path",
        help="Path for raw image data",
        default=None,
    )
    parser.add_argument(
        "--mni-template-path",
        help="Path to MNI templates, download the 'ICBM 2009c Nonlinear Symmetric' template in NIFTI format "
        "from https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)",
        default=None,
    )
    parser.add_argument(
        "--target-shape",
        help="3D image output shape, comma separated",
        default="256,256,256",
    )
    parser.add_argument(
        "--cores",
        help="Specify number of cores for parallel processing on current machine. Use more cores on larger machines with sufficient GPU memory",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--batch-range",
        help="Batch range as Python list indices.",
        default="0,-1",
    )
    parser.add_argument(
        "--download",
        help="Disable download of images via TorchIO",
        dest="download",
        default=False,
        action="store_false",
    )
    parser.add_argument("--raw-data-dir", help="Location of raw data", default=None)
    args = parser.parse_args()

    # setup software paths in environment var
    if "ants_dir" not in os.environ:
        if "CONDA_PREFIX" in os.environ:
            os.environ["ants_dir"] = os.path.join(os.environ["CONDA_PREFIX"], "bin")
        for executable in ["N4BiasFieldCorrection", "antsRegistration"]:
            if not os.path.exists(os.path.join(os.environ["ants_dir"], executable)):
                raise ValueError(
                    "Please set the environment variable 'ants_dir' to the location where ANTs is "
                    "installed (https://andysbrainbook.readthedocs.io/en/latest/ANTs/ANTs_Overview.html)"
                )

    intermediate_dir = args.intermediate_data if args.intermediate_data is not None else tempfile.mkdtemp()
    if args.raw_data_dir is None:
        args.raw_data_dir = os.path.join(intermediate_dir, "raw_images")
        os.makedirs(args.raw_data_dir, exist_ok=True)

    modalities = tuple(a.strip() for a in args.modalities.split(","))
    target_shape = tuple(int(a.strip()) for a in args.target_shape.split(","))
    batch_range = tuple(int(a.strip()) for a in args.batch_range.split(","))
    try:
        preprocessing_mri_pipeline(
            args.raw_data_dir,
            intermediate_dir,
            args.mni_template_path,
            args.output_dir[0],
            modalities,
            args.download,
            target_shape,
            batch_range,
            args.cores,
        )
    finally:
        if args.intermediate_data is None:
            shutil.rmtree(intermediate_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
