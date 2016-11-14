# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:51:39 2016

@author: Melvin
"""
#Implementation of MIQP Ramp loss SVM using IBM CPLEX 12.6.3 Python API
#Algorithm from : 
#J. Paul Brooks, (2011) Support Vector Machines with the Ramp Loss and the Hard Margin Loss. Operations Research 52(2):467-479
#Same notations as in paper

from __future__ import print_function
import numpy as np
import cplex
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import cross_validation
#Import following package to avoid scikit's deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from compiler.ast import flatten
#Compiler package is deprecated and removed in Python 3.x


#Labels preprocessing - all the target values need to be equal to 1 or -1
def PreprocessLabel(y_set):    
    for i in range(y_set.shape[0]):
        if y_set[i] == 0:
            y_set[i] = -1
    return y_set

#Soft normalization - subtract the mean of the values and divide by twice the standard deviation
def PreprocessData(X_set):
    for i in range(X_set.shape[0]):
        for j in range(X_set.shape[1]):
            X_set[i, j] = (X_set[i, j] - np.mean(X[j])) / 2 * np.std(X[j]) 
    return X_set        

#*******************************Dataset setting*******************************
#Simulated more or less noisy data
X, y = make_classification(n_samples= 100, n_features=3, n_redundant=0, n_informative=3,
                             n_clusters_per_class=2, random_state=123)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
       
PreprocessData(X)
PreprocessLabel(y)

#Use sklearn to split the dataset to a training set and a test set.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1238)        



#****************************Parameters setting********************************

#set_dual (bool) : Whether or not set the data to solve dual problem
#set_C (int/float) : Trade-off parameter
#set_kernel (string) : Positive semi-definite kernel = 'poly' for polynomial or 'rbf' for Radial Basis Function or 'linear' 
#set_degree (int) : Degree of polynomial kernel function
#set_gamma (int): Parameter of radial basis function
#set_localimplied(value): Instructs CPLEX whether or not to generate locally valid implied bound cuts for the model.
#                         value = -1 -> Do not generate locally valid implied bound cuts
#                         value = 0  -> Automatic: let CPLEX choose
#                         value = 1  -> Generate locally valid implied bound cuts moderately
#                         value = 2  -> Generate locally valid implied bound cuts aggressively
#                         value = 3  -> Generate locally valid implied bound cuts very aggressively

#set_timelimit: Sets the maximum CPU time, in seconds, for a call to CPLEX


set_kernel = 'linear'
set_C = 10
set_degree = 2
if set_kernel == 'linear':
    set_degree = 1
set_gamma = 1
set_dual = True
set_localimplied = 3
set_timelimit = 10


def MatrixToList(Mat):
    MatList = []
    for i in range(Mat.shape[1]):
        MatList.append([range(Mat.shape[1]), (Mat[i]).tolist()])
    return MatList


#Compute Gram matrix associed to chosen kernel : Kij = k(xi, xj)
# X_set = all the dataset to project in higher features space   
# degree = set degree for polynomial kernel
# gamma = set gamma parameter for rbf kernel
 
def Gram_Matrix(Kernel, X_set, Degree, Gamma):    
    print("Computing Gram Matrix...")
    Gram = np.zeros(shape = (X_set.shape[0], X_set.shape[0]))
    for i in range(0, (X_set.shape[0])):
        for j in range(0, (X_set.shape[0])):                
            if Kernel == 'poly':
                Gram[i, j] = polynomial_kernel(X_set[i], X_set[j], Degree)
            elif Kernel == 'rbf':
                Gram[i, j] = rbf_kernel(X_set[i], X_set[j], Gamma)
            elif Kernel == 'linear':
                Gram[i, j] = polynomial_kernel(X_set[i], X_set[j], Degree, coef0=0)
    #Use following instruction to fix Gram matrix symmetric problem  
    Gram = np.maximum(Gram, Gram.transpose())
    #Use following instruction to fix CPLEX Error 5002 (objective is not convex)
    if set_kernel == 'poly' or set_kernel == 'rbf':
        Gram = Gram + np.identity(Gram.shape[1])   
    print("Done")               
    return Gram

#FUNCTION : setproblemdata(Arguments)

#Arguments :
#Dual (bool) : Whether or not set the data to solve dual problem
#C (int/float) : Trade-off parameter
#kernel (string) : Positive semi-definite kernel = 'poly' for polynomial or 'rbf' for Radial Basis Function
#degree (int) : Degree of polynomial kernel function
#gamma (int): Parameter of radial basis function

#Parameters need to be tuned when calling the function by SVMIP1_RL or SVMIP2_RL

def setproblemdata(p, Dual=set_dual, C=set_C, kernel=set_kernel, degree=set_degree, gamma=set_gamma):
    
    if Dual == False:   
        
        print("Setting primal problem")

        p.set_problem_name("SVMIP1_RL")
    
        p.objective.set_sense(p.objective.sense.minimize)
    
        my_colnames = [["w" + str(i) for i in range(1, X_train.shape[1] + 1)], ["b"],
                        ["E" + str(i) for i in range(1, X_train.shape[0] + 1)],
                         ["z" + str(i) for i in range(1, X_train.shape[0] + 1)]]
    
        p.variables.add(types = [p.variables.type.continuous] * len(my_colnames[0]),
                        names = my_colnames[0], lb=[- cplex.infinity]*len(my_colnames[0]))
                    
        qmat = MatrixToList(np.identity(X_train.shape[1]))

        p.objective.set_quadratic(qmat)  
        
        p.variables.add(obj=[0], types = p.variables.type.continuous, names="b",
                        lb=[- cplex.infinity])
    
        p.variables.add(obj=[C] * len(my_colnames[2]),
                        types = [p.variables.type.continuous] * len(my_colnames[2]), names = my_colnames[2],
                        lb=[0] * len(my_colnames[2]), ub=[2] * len(my_colnames[2]))

        p.variables.add(obj=[2*C] * len(my_colnames[3]),
                    types = [p.variables.type.binary] * len(my_colnames[3]),
                    names = my_colnames[3])
    
        coefs = []
        for i in range(X_train.shape[0]):
            coefs.append([y_train[i] * X_train[i], y_train[i], 1.0])
            coefs[i][0] = coefs[i][0].tolist()
    
        wlist = my_colnames[0]
        Elist = my_colnames[2]
    
        for n in range(X_train.shape[0]):
            inds = flatten([wlist, "b", Elist[n]])
            fcoefs = flatten(coefs[n])
            p.indicator_constraints.add(indvar= my_colnames[3][n], complemented=1,
                                    rhs=1.0, sense='G',
                                    lin_expr=cplex.SparsePair(ind=inds, val=fcoefs))
                                    
                                    
    elif Dual == True:
        
        print("Setting dual problem")

        p.set_problem_name("SVMIP2_RL")
        
        p.objective.set_sense(p.objective.sense.minimize)
        
        my_colnames = [["a" + str(i) for i in range(1, X_train.shape[0] + 1)], ["b"],
                        ["E" + str(i) for i in range(1, X_train.shape[0] + 1)],
                         ["z" + str(i) for i in range(1, X_train.shape[0] + 1)]]
        
        p.variables.add(types = [p.variables.type.continuous] * len(my_colnames[0]),
                        names = my_colnames[0], lb = [0]* len(my_colnames[0]),
                        ub = [C]* len(my_colnames[0]))
        
        Kmat = Gram_Matrix(Kernel=kernel, X_set=X_train, Degree=degree, Gamma=set_gamma)
        Q = np.zeros(shape = (Kmat.shape[0], Kmat.shape[1]))
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                Q[i, j] = y_train[i] * y_train[j] * Kmat[i, j]
                
        qmat = MatrixToList(Q)   
        
        p.objective.set_quadratic(qmat)  
        
        p.variables.add(obj=[0], types = p.variables.type.continuous, names="b",
                        lb=[- cplex.infinity])

        p.variables.add(obj=[C] * len(my_colnames[2]),
                        types = [p.variables.type.continuous] * len(my_colnames[2]), names = my_colnames[2],
                        lb=[0] * len(my_colnames[2]), ub=[2] * len(my_colnames[2]))

        p.variables.add(obj=[2*C] * len(my_colnames[3]),
                        types = [p.variables.type.binary] * len(my_colnames[3]),
                        names = my_colnames[3])
        
        coefs = []  
        for i in range(X_train.shape[0]):
            coefs.append([y_train[i] * Kmat[i] * y_train, y_train[i], 1])
            coefs[i][0] = coefs[i][0].tolist()
            
        alist = my_colnames[0]    
        Elist = my_colnames[2]
        
        for n in range(X_train.shape[0]):
            inds = flatten([alist, "b", Elist[n]])
            fcoefs = flatten(coefs[n])
            p.indicator_constraints.add(indvar= my_colnames[3][n], complemented=1,
                                    rhs=1.0, sense='G',
                                    lin_expr=cplex.SparsePair(ind=inds, val=fcoefs))
            
def Predict(p, Test_set, label_test, Dual=set_dual):
    
    Test_set = X_test
    label_test = y_test
   
    sol = p.solution
    global test_predicted
    test_predicted = np.zeros(shape=label_test.shape[0])

    if Dual == False:
        
        sol_vals = []
        
        for i in range(X_train.shape[1] + 1):
            sol_vals.append(sol.get_values(i))
        
        w = np.asarray(sol_vals[0:len(sol_vals)-1])
        b = sol_vals[len(sol_vals)-1]
        
        for j in range(Test_set.shape[0]):
            test_predicted[j] = np.sign(np.inner(w, X_test[j]) + b)
    
    if Dual == True:
        sol_vals = []
        
        for i in range(X_train.shape[0]+1):
            sol_vals.append(sol.get_values(i))
        
        a = np.asarray(sol_vals[0:len(sol_vals)-1])
        b = sol_vals[len(sol_vals)-1]
        
        a_nonzero = []
        for i in range(len(a)):
            if a[i] != 0:
                a_nonzero.append([i, a[i]])
        if len(a_nonzero) == 0:
            print("No nonzero solution for dual variables")
        
        a_nonzero = np.asarray(a_nonzero)
        a_nonzero_index = a_nonzero[:, 0].astype(int)
        a_nonzero = a_nonzero[:,1]
                
        for i in range(a_nonzero.shape[0]):
            a_nonzero[i] = y_train[a_nonzero_index[i]] * a_nonzero[i] 
            
        kernel_mat = np.zeros(shape=(a_nonzero.shape[0], Test_set.shape[0]))
        X_critical = []
        
        for i in range(a_nonzero.shape[0]):
            X_critical.append(X_train[a_nonzero_index[i]])
        
        if set_kernel == 'poly':
            for i in range(len(X_critical)):
                for j in range(Test_set.shape[0]):
                    kernel_mat[i, j] = polynomial_kernel(X_critical[i], Test_set[j], set_degree)
        
        if set_kernel == 'rbf':
            for i in range(len(X_critical)):
                for j in range(Test_set.shape[0]):
                    kernel_mat[i, j] = rbf_kernel(X_critical[i], Test_set[j], set_gamma)
        
        if set_kernel == 'linear':
            for i in range(len(X_critical)):
                for j in range(Test_set.shape[0]):
                    kernel_mat[i, j] = polynomial_kernel(X_critical[i], Test_set[j], set_degree, coef0=0)
        
        for j in range(Test_set.shape[0]):
            test_predicted[j] = np.sign(np.inner(a_nonzero, kernel_mat[:,j]) + b)
        
    #Compute confusion matrix
    
    TP = np.zeros(shape=label_test.shape[0])
    TN = np.zeros(shape=label_test.shape[0])
    FP = np.zeros(shape=label_test.shape[0])
    FN = np.zeros(shape=label_test.shape[0])
    
    for i in range(label_test.shape[0]):
        if label_test[i] == 1 and test_predicted[i] == 1:
            TP[i] = 1
        elif label_test[i] == 1 and test_predicted[i] == -1:
            FN[i] = 1
        elif label_test[i] == -1 and test_predicted[i] == 1:    
            FP[i] = 1
        elif label_test[i] == -1 and test_predicted[i] == -1:
            TN[i] = 1
    
    Confusion_matrix = [[np.sum(TP), np.sum(FN)], [np.sum(FP), np.sum(TN)]]        
    print("Confusion matrix = ([TP, FN], [FP, TN]) = ", Confusion_matrix)


    Sensitivity = Confusion_matrix[0][0] / (Confusion_matrix[0][0] + Confusion_matrix[0][1])
    Precision = Confusion_matrix[0][0] / (Confusion_matrix[0][0] + Confusion_matrix[1][0])
    Accuracy = (Confusion_matrix[0][0] + Confusion_matrix[1][1]) / (Confusion_matrix[0][0] + 
                Confusion_matrix[1][1] + Confusion_matrix[0][1] + Confusion_matrix[1][0])   
    print("Classifier Accuracy = ", Accuracy )
    print("Precision = ", Precision)
    print("Sensitivity = ", Sensitivity)
           
    return test_predicted
 
 
def SVMIP1_RL():
    
    p = cplex.Cplex()
    setproblemdata(p, Dual=False)
    
    p.write("SVMIP1_RL.lp")    

    p.parameters.timelimit.set(set_timelimit)
    p.parameters.mip.cuts.localimplied.set(set_localimplied)
    
    print("Solving Ramp Loss SVM primal problem")

    p.solve()
    
    sol = p.solution
   
    sol.write("Primal_Solution.lp")

    # solution.get_status() returns an integer code
    print("Solution status = ", sol.get_status(), ":", end=' ')
    
    # the following line prints the corresponding string
    print(sol.status[sol.get_status()])
    print("Solution value  = ", sol.get_objective_value())

    numcols = p.variables.get_num()

    for j in range(numcols):
        print("Column %d: Value = %10f" % (j, sol.get_values(j)))

        
    print("Test set accuracy")
    Y_pred = Predict(p, X_test, y_test)


def SVMIP2_RL():
    
    p = cplex.Cplex()
    setproblemdata(p, Dual=True)
    
    p.parameters.timelimit.set(set_timelimit)
    p.parameters.mip.cuts.localimplied.set(set_localimplied)

    p.write("SVMIP2_RL.lp")    
    
    print("Solving Ramp Loss SVM dual problem")

    p.solve()
    
    sol = p.solution
    
    sol.write("Dual_Solution.lp")

    # solution.get_status() returns an integer code
    print("Solution status = ", sol.get_status(), ":", end=' ')
    
    # the following line prints the corresponding string
    print(sol.status[sol.get_status()])
    print("Solution value  = ", sol.get_objective_value())

    numcols = p.variables.get_num()

    for j in range(numcols):
        print("Column %d: Value = %10f" % (j, sol.get_values(j)))
    

    print("Test set accuracy")
    Y_pred = Predict(p, X_test, y_test)

             

if __name__ == "__main__" and set_dual==False:
    SVMIP1_RL()   
elif __name__ == "__main__" and set_dual==True:
    SVMIP2_RL()
else:
    print("Error: set Dual value to True or False to run the program")