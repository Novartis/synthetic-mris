#!/bin/bash

# shellcheck disable=SC2005
echo "$(pwd)"

set -e
mkdir -p conda-env-preprocessing
if [[ ! -f "conda-env-preprocessing/bin/python3.10" ]]; then
    echo "Creating new conda env"
    rm -rf conda-env-preprocessing
    conda create --no-default-packages -p conda-env-preprocessing -y -c conda-forge python=3.10
else
    echo "Keeping current conda env. To re-create, run"
    echo "rm -rf $(pwd)/conda-env-preprocessing"
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate ./conda-env-preprocessing &> /dev/null || source activate ./conda-env-preprocessing

CONDACMD=$(command -v mamba || command -v conda)
if [[ -d "/usr/local/cuda/bin" ]]; then
    # Run on a GPU node to ensure nvcc is available
    export PATH=/usr/local/cuda/bin:$PATH
    ${CONDACMD} install -y -c conda-forge -c pytorch -c nvidia \
        pytorch=2.1 pytorch-lightning=2.1 pytorch-cuda=12.1 torchvision=0.16 torchio=0.19 \
        pandas=2.1 seaborn=0.13 scikit-learn=1.7 umap-learn=0.5 \
        loguru=0.7 xlrd=2.0 openpyxl=3.1 pyarrow=14.0 \
        hydra-core=1.3 \
        ants=2.5 scikit-image=0.22
else
    ${CONDACMD} install -y -c conda-forge -c pytorch \
        pytorch=2.1 pytorch-lightning=2.1 torchvision=0.16 torchio=0.19 \
        pandas=2.1 seaborn=0.13 scikit-learn=1.7 umap-learn=0.5 \
        loguru=0.7 xlrd=2.0 openpyxl=3.1 pyarrow=14.0 \
        hydra-core=1.3 \
        ants=2.5 scikit-image=0.22
fi

pip install git+https://github.com/MIC-DKFZ/HD-BET.git@ae160681324d524db3578e4135bf781f8206e146