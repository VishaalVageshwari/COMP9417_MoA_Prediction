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

Create a notebook for the competition and upload the relevant notebook from the notebooks directory (tabnet_and_simple.ipynb has both models). The competition requires that the notebook be run offline but there are certain non-standard libaries that are required. Add the following two libraries to the kaggle input:

1. https://www.kaggle.com/yasufuminakama/iterative-stratification

2. https://www.kaggle.com/ryati131457/pytorchtabnet

Note: The baseline neural network only requires the first data library.

Safe that version and submit it to confirm kaggle scores.

Change the line 
```python
sub = run_net('tab', 'cv')
```
to
```python
sub = run_net('simple', 'cv')
```
for the baseline model.

Results vary slightly between the two implementations (local vs kaggle) despite being identically seeded (speculated to be because of os differences). The kaggle notebook metrics were taken for consistency.