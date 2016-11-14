# RSVM
Implementation of Ramp Loss Support Vector Machine using IBM CPLEX 12.6.3

Introduced by Vapnik and Cortes in 1995, SVM is an optimization based supervised classification method for finding 
a hyperplane that separates two classes of observations with interesting generalization abilities and good statistical properties.
A non-trivial way to produce non-linear decision boundaries is to use the so-called Kernel Trick to project data into a higher dimensional space. 
In practice, traditional Hinge Loss SVM performs poorly in presence of outliers (i.e. observation point that is distant from other observations) 
because of their influence on the separating hyperplane. J-P Brooks (2011) proposed new SVM formulations using another measure of classification 
error resulting to an increased robustness to outliers. 
Ramp Loss SVM and Hard Margin Loss SVM are formulated as Quadratic Mixed Integer Programs and implemented using CPLEX 12.6.3 Python API.

This implementation is part of my first research project in Machine Learning. 

Scikit learn's features are used for artificial dataset creation and kernel functions computing, Type A and Type B outliers (see J-P Brooks for description) could be added 
to test performances of classifiers on contaminated data. 

Linear, Polynomial and RBF kernel are implemented. See : http://scikit-learn.org/stable/modules/svm.html#svm-kernels

Solving time could impact computed gap remaining after solving and hence the feasible solution. 
It and can be tuned using "set_timelimit" variable in seconds. 
I recommend to set 10s solving time for cross-validation and then train the best performing model with a larger time limit.  

Solving primal program could be computationaly expensive for large training set. 





References :

Cortes, C. and V. Vapnik, « Support-vector networks », Machine Learning, 20(3), p. 273–297, 1995.

J. Paul Brooks, Support Vector Machines with the Ramp Loss and the Hard Margin Loss. Operations Research 52(2):467-479, 2011.

Eric J. Hess and J. Paul Brooks, The Support Vector Machine and Mixed Integer Linear Programming: Ramp Loss SVM with L1-Norm Regularization,
14th INFORMS Computing Society Conference Richmond, Virginia, January 11–13, 2015 pp. 226–235






For personnal use only.
