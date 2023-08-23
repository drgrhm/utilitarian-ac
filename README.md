# utilitarian-configuration

Download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html).

Unpack it into a folder called icar/.

Setup directories: 
```
mkdir dat img
```

## Run Example: 

Run the Naive algorithm with a Pareto(10, 1) utility function epsilon = .2, delta=.1 and captime = 100: 
```
python naive.py minisat u_pareto-10-1 .2 .1 100
```
