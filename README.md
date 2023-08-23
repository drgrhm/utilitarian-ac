# utilitarian-algorithm-configuration

Code to reproduce experiments from the paper [Utilitarian Algorithm Configuration](https://www.cs.ubc.ca/~drgraham/datasets.html).

Download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder called `icar/`.

Set up directories: 
```
mkdir dat img
```

Run the Naive algorithm for different captimes and epsilons (Figure 1):
```
python naive_captime_experiement.py [minisat | cplex_rcw | cplex_region]
```


Compare total runtime of the Naive algorithm and UP for different epsilons (Figures 2, 3):
```
python anytime_speedup_experiement.py [minisat | cplex_rcw | cplex_region]
```


Plot cdfs of the dataset and utility functions:
```
python data_explore.py [minisat | cplex_rcw | cplex_region]
```
