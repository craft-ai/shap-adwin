# Code and paper for

## Setup

Use poetry python package manager to install the correct dependencies :

- [Python](https://www.python.org) (any version >3.8 should work),
- [Poetry](https://python-poetry.org).

Once poetry is installed, in order to install the dependencies run:

 ```console
  poetry install
  ```

Many functions rely on a file path to load and write data. In order for the code to work properly set the `my_path` variable to mach the code directory unzipped so that in your local directory you see the following files:

```
src
data
README.md
pyproject.toml
poetry.lock
```

To run the notebook :

 ```console
  poetry shell
  jupyter notebook
  ```

### Data Directory

Contains several drift dataset along with already generated drift results.
`./data/results` is a directory where each sub directory contains the performance/drift-results of a given Drift for various Drift-detection heuristics.

For instance to find results for a `Brutal concept drift back and forth` and for the `SHAP ADWIN` heuristic go to : `./data/results/brutal_concept_backforth/shap_adwin` and you will find pickle files of several runs results.

### Important files

- `./src/PLOT- Paper PLOTS.ipynb` : generates the plots used in the paper

- `./src/drift_generators.py` : Contains the main code to generate the different Drifts ; Drift metrics ; Plot functions ; some retrain functions (not used for the paper) ; Loading function to load results ;
  
:construction: The path is hard coded here, you need to modify it `my_path = /home/...`

- `./src/RUN - Paper results.ipynb` : generates results for the many drift cases presented in the paper or available in drift_generator.py and store model error + detection inidices in a pickle file.
  
:construction: The path is hard coded here, you need to modify it `my_path = /home/...`

- `./src/PLOTS - complete overview.ipynb` : A notebook going through many drift cases and exposing the drift-detectors performances in various plots.
    ! beware this notebook tends to be exhaustive so it takes a while to run !
  
- `./src/SHAP values around drift point.ipynb` : Explores the scattering of shapley values arround the drift point.

:construction: BEWARE some notebooks can take a lot of time to run. Be sure to run each cell carefully (notably the run - Paper results notebook do not run every cell or adjust the `n_iter` variable carefully.)

### General Advice

- Use a table of content extension to navigate the notebooks. They are quite lengthy and most of the time you only need to access a section as most sections isolated with proper markdowns cells are idependant you can go there and exectute the partial group of cells.
