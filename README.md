# Graduate Program ML Exercises

This repository contains my solutions to the coding exercises for the Graduate Program [M2A](https://m2a.lip6.fr/) at Sorbonne University.

Each folder corresponds to one course:
+ `AMAL` contains the material for the deep learning course
+ `BDLE` contains the material for the large-scale database course
+ `reinforcement` contains the material for the reinforcement learning course
+ `intro_ML` contains the material for the course on foundations of Machine Learning

More material for each course will be added progressively, and some new courses may be added later.

For now, the material is randomly in French or in English, but I'll try translating everything into English at some point.

For each course, inside the corresponding folder, there is a `environment.yml` file describing the conda environment required to run the material (mostly Jupyter notebooks).

To create the environment, run 

``` bash
conda env create -f environment.yml
```

For more information on conda environments, please visit the [conda official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).