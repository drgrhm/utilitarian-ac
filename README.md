# utilitarian-ac

Code to reproduce experiments from the paper [Utilitarian Algorithm Configuration](https://www.cs.ubc.ca/~drgraham/datasets.html).

Execute the following to download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder named `icar/`:
```
mkdir icar
wget https://www.cs.ubc.ca/~drgraham/datasets/dataset_icar.zip
unzip dataset_icar.zip -d icar/
```

Set up directories: 
```
mkdir dat img
```

Run the Naive algorithm for different captimes and epsilons (Figure 1):
```
python naive_captime_experiment.py [minisat | cplex_rcw | cplex_region]
```


Compare total runtime of the Naive algorithm and UP for different epsilons (Figures 2, 3):
```
python anytime_speedup_experiment.py [minisat | cplex_rcw | cplex_region]
```


Plot empirical CDFs of the dataset and utility functions (not included in paper):
```
python data_explore.py [minisat | cplex_rcw | cplex_region]
```
