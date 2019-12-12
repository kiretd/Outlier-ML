# Outlier-ML
Demo scripts for machine learning outlier detection methods using scikit-learn.

Demo scripts are provided for outlier and anomaly detection using the following methods:
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [One-Class Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)

## Dependancies
*  python
* [NumPy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/index.html)

## Usage
Each file is meant to be run as a script, either in Spyder or using the following commands:
```python
python IsolationForest.py
python LocalOutlierFactor.py
python OneclassSVM.py
```
However, using an IDE with a workspace like Spyder is ideal, 
because following the comments and running the code line by line
will allow you derive the most benefit from the way these demos
are structured!

## Outputs
The demo scripts will produce console outputs and plots. 
Please check that you get nearly identical plots as these 
to confirm successful operation of the demos on your machine
and python configuration.

### Isolation Forest
Isolation Forest decision function (for contamination proportion of 0.4) and performance for difference contamination proportions.
![Isolation Forest Decision Function][isofor_df]
![Isolation Forest Performance][isofor_perf]

### One-Class Support Vector Machine
Visualization of the One-Class SVM decision function.
![SVM Decision Function][svm_df]

### Local Outlier Factor
Visualization of LOF output: sample data with Outlier Score for each sample. 
Larger score means high probability of being an outlier.
![LOF Output Visualization][lof_viz]

[isofor_df]: https://github.com/kiretd/Outlier-ML/blob/master/sample_images/IsoFor_map.png
[isofor_perf]: https://github.com/kiretd/Outlier-ML/blob/master/sample_images/IsoFor_perf.png
[lof_viz]: https://github.com/kiretd/Outlier-ML/blob/master/sample_images/LOF_visualize.png
[svm_df]: https://github.com/kiretd/Outlier-ML/blob/master/sample_images/SVM_map.png

## Authors
Kiret Dhindsa
