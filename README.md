# BLEanalysis: finley-dissertation

_Date: 26th April 2026_

This repository branches off of `BLEanalysis:main` and contains the code and Jupyter notebooks used
for Finley's (my) undergraduate dissertation project. The project looks at the feasability of powering the Bluetooth
LE bee tags with gallium arsenide solar cells rather than supercapacitors. Due to the decreased power output of solar
cells, the path inference algorithm must sample less frequently, meaning the affect of attenuation on the signal
cannot be assumed to be the same and integrated out. Therefore, the attenuation must be modelled so the path inference
algorithm can take it into account. PyMC is used to attempt to fit a variety of distributions onto the data. Using
PyMC's powerful paramter inference abilities, it can discover the parameters that best fit a probability distribution
onto the data. Not all distributions will accurately fit the data, so the "best" parameters can only do so much!

All notebooks are under `/jupyter/Finley/`. The most
important notebooks are `FSPL.ipynb` and `RSS Differences.ipynb`, with the others used for getting familiar
with the BeeLE project and experimenting with the existing code and libraries. 

The `BleFinley` supporting library, under `/BleFinley/`, was created to parse the raw data from the bee tags
and GPS receiver so data analysis could be conducted on them. The library focuses heavily on semantics and doesn't
consider performance. Only ideas relevant to my dissertation have been coded, nothing else (including the path
reconstruction algorithms). Supporting functions for the `FSPL.ipynb` and `RSS Differences.ipynb` notebooks are
defined in files of the same name. 

The Bluetooth logs and GPS data that is analysed by the two main notebooks is from the experiments called
`March 26 2025 Field Trial`. The other, unimportant notebooks also look at data from the `Feb 18 2025 Field Trial` but
not in any serious detail. None of this data was not collected by me.

## Installation

Python 3.12 is required for this project. Please ensure it is installed before running the commands below. The commands
are in bash and assume you are on Linux or WSL.

All dependencies for the project have been defined in the `requirements.txt` file, including the very specific 
versions of some libraries required for path inference. To clone the git repository and install these dependencies with
the two custom libraries in editable mode, run the following:

```bash
git clone git@github.com:SheffieldMLtracking/BLEanalysis.git
cd BLEanalysis
git checkout finley-dissertation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip installe -e . # install the BLEanalysis package stored in the root directory
pip install -e BleFinley
```

### Experiment Data

The data from past field experiments must be downloaded too. They are accessed via a sim-link to a repository stored
in this repository's parent direction (`../bluetooth_experiments`). That repository is private, and you must be added
by someone with the authority to do so. Assuming you have gained access, run the following:

```bash
git clone git@github.com:SheffieldMLtracking/bluetooth_experiments.git ../bluetooth_experiments
```