# COMP9417_MoA_Prediction

### Step 1: Download kaggle MOA competition dataset

The kaggle competition link can be found here: https://www.kaggle.com/c/lish-moa

Download the competition data set and place the lish-moa directory inside the data directory.

### Step 2: Install all relevant python libaries

All the relevant libraries are given in requirements.txt and can be installed through pip as follows:

```console
pip install -r requirements.txt
```

### Step 3: Change directories to src

The scripts to train the models and collect the relevant ouput are in the src directory.

```console
cd src
```

### Step 3: Run models

To run cross-validation and subsequent prediction on the test set using the Simple baseline model run the following command:

```console
python3 run.py --name simple
```

To run cross-validation and subsequent prediction on the test set using the TabNet model run the following command:

```console
python3 run.py
```
or
```console
python3 run.py --name tab
```

This may take a while to complete depending on the hardware avaiable.

## Running on kaggle

### Upload notebook

Create a notebook for the competition and upload the relevant notebook from the notebooks directory (tabnet_current_best.ipynb is the main model). 

Safe that version and submit it to confirm kaggle scores.