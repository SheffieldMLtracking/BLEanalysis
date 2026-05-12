# BLEanalysis: finley-dissertation

_Date: 12th May 2026_

This repository branches off of `BLEanalysis:main` and contains the code, Jupyter notebooks, and experiment data used
for Finley's (my) undergraduate dissertation project. Other code, notebooks and experiments by Chis and Michael have been removed
prior to the dissertation handin deadline. 

The project looks at the feasibility of powering the Bluetooth
LE bee tags with gallium arsenide solar cells rather than supercapacitors. Due to the decreased power output of solar
cells, the path inference algorithm must sample less frequently, meaning the affect of attenuation on the signal
cannot be assumed to be the same and integrated out. Therefore, the attenuation must be modelled so the path inference
algorithm can take it into account. PyMC is used to attempt to fit a variety of distributions onto the data. Using
PyMC's powerful paramter inference abilities, it can discover the parameters that best fit a probability distribution
onto the data. Not all distributions will accurately fit the data, so the "best" parameters can only do so much!

## Repository Structure

- `/jupyter/`
  - All notebooks are stored here. The most important notebooks are `FSPL.ipynb` and `RSS Differences.ipynb`, with 
        the others used for getting familiar and experimenting with the project and other inference libraries
- `/BleFinley`
  - The `BleFinley` supporting library was created to parse the raw data from the bee tags
        and GPS receiver so data analysis could be conducted on them. The library heavily focuses on semantics and 
        doesn't consider performance. Only ideas relevant to my dissertation have been coded, nothing else 
        (including the path reconstruction algorithms). Supporting functions for the `FSPL.ipynb` and 
        `RSS Differences.ipynb` notebooks are defined in files of the same name.
- `/bluetooth_experiments`
  - Two sets of experimental data are analysed, originating from the private 
        [bluetooth_experiments repository](https://github.com/SheffieldMLtracking/bluetooth_experiments). All data
        used has been copied into this repository to allow my code to run. None of the data was collected by me.

## Installation

Python 3.12 is required for this project.

All dependencies for the project have been defined in the `requirements.txt` file, including the `BLEanalysis` library. 
To clone the git repository and install dependencies (including `BleFinley`) on Linux/WSL, run the following:

```bash
git clone git@github.com:SheffieldMLtracking/BLEanalysis.git
cd BLEanalysis
git checkout finley-dissertation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```