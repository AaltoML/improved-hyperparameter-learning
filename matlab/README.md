# Matlab baselines

The codes have been tested in Mathworks Matlab R2021a. We used Matlab (The MathWorks, Inc.) R2021a.
To set up the environment, download the following toolboxes:
* [GPML toolbox v. 4.2](http://gaussianprocess.org/gpml/code/matlab/doc/) 
* [GPstuff toolbox](https://github.com/gpstuff-dev/gpstuff/tree/c7d797d8ff09da845613165b0f94030e36d501e5)

After downloading the toolboxes, modify the paths in the Matlab scripts accordingly.

## Marginal likelihood contour surface

The code in the `contour-plots` folder uses the GPML toolbox for MCMC estimation of marginal likelihood surface. The scripts `run_lml.sh` and `run_lp.sh` were used for submitting the jobs for log marginal likelihood estimation and log predictive density estimation. These scripts produce the results for the `ionosphere` data set (leftmost panels in Fig. 2 in the paper). 

Surface contour plot jobs for the other data sets can be submitted by changing the data set name in the bash scripts.

## MCMC results for classification

The code in the `classification` folder uses the GPStuff toolbox for MCMC log predictive density estimation in classification. The script `run_mcmc_classification.sh` was used for submitting the jobs.

To run these scripts, we use the same splits as in Python. To ensure the splits to be the same, we export the pre-split data and write it on disk. These data-splits can be created by running

```bash
python generate_matlab_data.py
```

in the `python/experiments/classification` folder (found in the root of this repository). Alternatively, the generated splits can be downloaded as mat files from [Google Drive](https://drive.google.com/file/d/1hhx3wzdDtTmpKgdb6isPr5EJ7lDO8UeN/view?usp=sharing) as a zip file. Unzip the file inside the `classification` folder.

## LA, EP, and MCMC results for Student-t regression

The code in the `student-t` folder uses the GPStuff toolbox for LA, EP and MCMC log predictive density estimation in Student-t regression. The script `run_mcmc_student_t.sh` was used for submitting the jobs.
