IRP Artifacts

File Structure

To Recreate the Independent Research Project, the contents of this .zip file must be saved in the user's University of York Google Drive in "/content/drive/MyDrive/ResearchProject".

.ipynb files
Where necessary, .ipynb files require mounting the users google drive to access the file structure. This will be carried out in the first cell with the following commands:

---

from google.colab import drive
drive.mount('/content/drive/')

import os
os.chdir("/content/drive/MyDrive/ResearchProject")

---

File irp_functions.py

This is a python file containing functionality to implement experiments. It is duplicated in every folder containing a .ipynb file which imports its functionality as below:

---

from irp_functions import *

---

Contents of the File System

MyDrive/ResearchProject

    - Folders: Benchmarking, Datasets, Ethics, Results, Synthetic_Datasets
    - Notebooks: Abalone.ipynb, Ecoli.ipynb, Mammography.ipynb, Wine.ipynb, 
        Yeast.ipynb all load named datasets and run experiments for all oversamplers and rates before saving results to file.
    - Notebooks: COVID_1.ipynb to COVID_11.ipynb load the COVID dataset and run required experiments. Separate files were required to prevent timeout due to long experiments
    - irp_functions.py - Python file containing functionality for required experiments

MyDrive/ResearchProject/Benchmarking

    - .csv files showing output of Benchmarking experiments with imbalance-learn SMOTE implementation to establish internal validity as discussed in report
    - .ipynb files causing above results to be produced, one for each dataset.
    - .xlsx files used for analysis of results
    - .irp_functions for required experiment functionality


MyDrive/ResearchProject/Datasets

    - abalone.csv - abalone dataset
    - ecoli.csv - ecoli dataset
    - full_einstein_25col.csv - Covid Dataset
    - mammography.csv - mammography dataset
    - wine_quality.csv - wine dataset
    - yeast.csv - yeast dataset

MyDrive/ResearchProject/Ethics

    - emails and attachments relating to:
        - original ethics application (5 UCI datasets approved, Einstein dataset as used by Turlapati & Prusty rejected).
        - supplemental ethics application (new COVID dataset from Einstein Hospital approved).

MyDrive/ResearchProject/Results

    - Folders: Abalone, Covid, Ecoli, Mammography, Wine, Yeast - each contains:
        - 41 csv files which represent experimental results after application of no oversampler (benchmark.csv) and then applications of ten oversampling rates to Outlier_SMOTE with three different distance metrics and to SMOTE.
        - one .xlsx file used to calculate sum and average experimental outputs as described in the report.
    - AUROC.csv - file of AUROC scores for all experiments and datastes for statistical testing.
    - f1.csv - file of F1 scores for all experiments and datastes for statistical testing.
    - Wilcoxon.ipynb - Notebook file which was used to perform statistical tests on above csv files.

MyDrive/ResearchProject/Synthetic_Datasets

    - Folders: Abalone, Covid, Ecoli, Mammography, Wine, Yeast - each contains synthetic datasets as created by each fold of each experiment set for the six datasets, created to establish internal validity. Naming convention: DatasetName-OversamplingAlgorithm-DistanceMetric-OversamplingRate-ExperimentNumber-FoldNumber.csv
    - .ipynb files required to create csvs as described above.
    - irp_functions.py for experimental functionality










    
