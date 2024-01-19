Public repository for "Shapley-Detect: A Novel Approach for Robust Drift Detection in Multivariate Dynamic Environments"

## Setup

Use poetry python package manager to install the correct dependencies :

- [Python](https://www.python.org) (any version >3.8 should work),

 ```console
  python3 -m venv ./venv
  source venv/bin/activate
  python3 -m pip install -r requirements.txt
  ```

Then you will need to edit a .env file with by specifying the following path variables (replace YOUR_PATH by your current working directory):

 ```
  RESULTS_ROOT_PATH="YOUR_PATH/Shap-Adwin/results/"
  FIGURES_PATH="YOUR_PATH/Shap-Adwin/figures/"
  DATA_PATH="YOUR_PATHShap-Adwin/data/"
  ```


## Important files

- `./src/notebooks/bench.ipynb` : generates the results used in the paper;

- `./src/notebooks/display_results.ipynb` : generates the figures and results vizualisations used in the paper ;

## Add new dataset

To add a new dataset place your dataset in .csv file in the /data/ and then do exactly the same as with sine1 or sine2 or stagger case. 

## General Advice

- Use a table of content extension to navigate the notebooks. They are quite lengthy and most of the time you only need to access a section s most sections isolated with proper markdowns cells are idependant you can go there and exectute the partial group of cells.
- To run the notebook make sure you are using the virtual env previously created with the correct dependencies installed.

