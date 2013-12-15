#! /usr/bin/env python

import numpy as np
import pylab as pl

from scipy import optimize as opt


def load_yeast2_data( fname ):
    """

    Function:  LOAD_YEAST2_DATA( fname )

    Description: load the data in from the GSE series matrix and parse

    Parameters: fname -- name of the .txt file of time series data 

    """
    # We read in the series matrix data and get rid 
    # of the junk we don't want.
    fh = open(fname, 'r')
    gse = fh.read().split('\n')
    fh.close()
    K_begin = 71
    K_end = -2
    series = gse[K_begin:K_end]
    # We create a dictionary out of the mapping from 
    # Affymetrix IDs to ENSEMBL IDs    
    fh = open('data/y2E.dat', 'r')
    y2e = fh.read().split('\n')[1:-1]
    fh.close()
    y2e_dict = {}
    for i in range(len(y2e)):
        y = y2e[i].split('\t')
        y2e_dict[y[0][1:-1]] = y[1][1:-1]
    # We want a time series matrix for the expression levels of the genome
    # of S. cerevisiae only.
    cerevisiae = []
    gene_lst = []
    for i in range(len(series)):
        y = series[i].split('\t')
        affy = y[0][1:-1]
        if affy in y2e_dict.keys():
            gene = '"'+y2e_dict[affy]+'"'
            cerevisiae.append([gene]+y[1:])
            gene_lst.append(gene[1:-1])
    # Having filtered out the rows for the organism S. cerevisiae, we
    # create a new time series matrix now that we know how many Affymetrix
    # IDs we can map onto the ENSEMBL database.
    K_cerevisiae = len(cerevisiae)
    K_time = len(y[1:])
    cerevisiae_mat = np.zeros((K_cerevisiae,K_time))
    for i in range(len(cerevisiae)):
        cerevisiae_mat[i,:] = np.array(cerevisiae[i][1:])
    
    return [cerevisiae_mat, gene_lst]

class dataDynamics:
    """

    Class dataDynamics -- Find optimal linear models for reduced order
    systems of time series data.

    """
    def __init__(self, fname,nCV, nTest, nPrune=0):
        """

        Method:  __INIT__(self, fname, nCV, nTest, nPrune=0) 

        Description: Initialize values for the class dataDynamics.  

        Parameters: fname -- name of GSE*_series_matrix.txt file
                    nCV -- number of cross-validation samples
                    nTest -- number of samples for test case
                    nPrune=0 -- cut off the first nPrune samples

        """
        # Store data in the class for use by its methods.
        self.file = fname
        self.nCV = nCV
        self.nTest = nTest
        self.nPrune = nPrune
        # Load in yeast2 data.
        self.tSeries, self.geneLst = load_yeast2_data(fname)
        # Create training features X, Xcv, and Xtest.
        self.X = self.tSeries[:,:-(nCV+nTest+1)]
        self.Xcv = self.tSeries[:,-(nCV+nTest+1):-nTest-1]
        self.Xtest = self.tSeries[:,-nTest-1:-1]
        # Create training targets y, ycv, and ytest.
        self.y = self.tSeries[:,1:-(nCV+nTest)]
        self.ycv = self.tSeries[:,-(nCV+nTest):-nTest]
        self.ytest = self.tSeries[:,-nTest:]
        self.nFeatures = self.X.shape[0]
        self.nExamples = self.X.shape[1]
        # Set a couple of flags to False for use later by
        # featureNormalize() and pca().
        self.zFlag = False
        self.pcaFlag = False

    def featureNormalize(self):
        """

        Method:  FEATURENORMALIZE(self)

        Description: Normalizes the features of class dataDynamics,
        excluding the member of the cv and test sets.

        Parameters: passed through to method from __init__.

        """
        # Don't normalize the features of the CV and test sets.
        tS = self.tSeries[:,:-(self.nCV+self.nTest)]
        nFeatures, nExamples = tS.shape
        # Find the mean (mu) and norm (sigma) of the features.
        mu = np.mean(tS,axis=1)
        sigma = np.std(tS, axis=1)
        # Vectorized implementation of (x-mu)/sigma.
        MU = np.tile(mu,(nExamples,1)).T
        SIGMA = np.tile(sigma,(nExamples,1)).T
        zScore = (tS-MU)/SIGMA
        # Attach the data in the variables to the class.
        self.mu = mu
        self.sigma = sigma
        self.zScore = zScore
        self.zFlag = True
        # Use mu and sigma to transform the CV set to the
        # same coordinates as the normalized training features.
        SIGMA_CV = np.tile(sigma,(self.nCV,1)).T
        MU_CV = np.tile(mu,(self.nCV,1)).T
        self.zScoreXcv = (self.Xcv - MU_CV)/SIGMA_CV
        self.zScoreycv = (self.ycv - MU_CV)/SIGMA_CV
        # Use mu and sigma to transform the test set to the
        # same coordinates as the normalized training features.
        SIGMA_TEST = np.tile(sigma,(self.nTest,1)).T
        MU_TEST = np.tile(mu,(self.nTest,1)).T
        self.zScoreXtest = (self.Xtest - MU_TEST)/SIGMA_TEST
        self.zScoreytest = (self.ytest - MU_TEST)/SIGMA_TEST

    def pca(self):
        """

        Method: PCA(self)
        
        Description:  performs principal component analysis of the 
        normalized features from the time series data via the SVD.

        Parameters:  passed through from __init__ and featureNormalize()

        """
        # Check that the features have been normalized.
        if not self.zFlag:
            self.featureNormalize()
        z = self.zScore
        # SVD the zScores
        u, s, v = np.linalg.svd(z)
        self.U = u
        self.S = s
        self.V = v
        # Let the other methods know U, S, and V are now available.
        self.pcaFlag = True

    def randInitTheta(self,init_epsilon):
        """

        Method:  RANDINITTHETA(self, init_epsilon)
        
        Description: create a vector of random numbers contained 
        inside the interval (-init_epsilon,+init_epsilon).

        Parameters:  init_epsilon -- the lower and upper bound
                     passed in through method fit_reduced()

        """
        self.theta = 2*init_epsilon * \
            np.random.rand(self.nModes**2,) - \
            init_epsilon
        return self.theta

    def regCostFunction(self, theta, X, y, lam):
        """

        Method: REGCOSTFUNCTION(self, theta, X, y, lam)

        Description: objective function to be minimized to find
        optimal discrete-time linear dynamical model to the
        training features X and the training targets y with 
        regularization parameter lam.

        Parameters:  theta -- nModes X nModes transition matrix
                     X -- training features
                     y -- training targets
                     lam -- regularization parameter

        """
        # nModes is pulled out from the class.
        # "This is the beauty of OOP. I can pull variables in
        #  from the class when I know a method is only going
        #  to be called by another method that assigns 
        #  self.nModes with the correct value." -- observation
        nModes =self.nModes
        Theta = theta.reshape((nModes,nModes))
        return np.sum((np.dot(Theta,X) - y)**2 \
                          + lam*np.linalg.norm(Theta,ord=1)) \
                          /(2*len(X.ravel()))

    def fit_reduced(self, nModes, nTries, lam, nSamples=None):
        """

        Method: FIT_REDUCED(self, nModes, nTries, lam, nSamples=None)

        Description: A method to fit a linear, discrete-time dynamical
        model to a set of data contained in the class dataDynamics.

        Parameters: nModes -- the degree of the first order system
                              used to model the data
                    nTries -- the number of times to train a model
                    lam -- the regularization parameter
                    nSamples -- the number of samples to include in 
                                the training set.
       
        """
        # If we don't use the nSamples variable, we use the 
        # maximum number of examples in the training set and
        # get rid of the first nPrune samples.
        if nSamples==None:
            nSamples = self.V.shape[1]-self.nPrune
        # We need the reduced order modes to fit the dynamics.
        if self.pcaFlag == False:
            self.pca()
        # This is passed through to regCostFunction.
        self.nModes = nModes
        self.lam = lam
        # Transposing a matrix is expensive. Do this once and
        # reference uT instead of transposing U many times.
        uT = self.U.T
        # Create a training set.
        self.Vtrain = self.V[:nModes,self.nPrune:self.nPrune+nSamples]
        # Create cv and test sets by rotating by U and scaling by S.
        self.Vxcv = np.multiply(
            1./self.S[:nModes],
            np.dot(uT[:nModes,:], self.zScoreXcv).T).T
        self.Vxtest = np.multiply(
            1./self.S[:nModes],
            np.dot(uT[:nModes,:],self.zScoreXtest).T).T
        self.Vycv = np.multiply(
            1./self.S[:nModes],
            np.dot(uT[:nModes,:], self.zScoreycv).T).T
        self.Vytest = np.multiply(
            1./self.S[:nModes],
            np.dot(uT[:nModes,:], self.zScoreytest).T).T

        # Create training features and training targets
        Vx = self.Vtrain[:,:-1]
        Vy = self.Vtrain[:,1:]
        # Scramble the sample features and targets together.
        randV = np.random.permutation(
            np.concatenate((Vx,Vy),axis=0).T 
            ).T
        self.Vx = randV[:nModes,:]
        self.Vy = randV[nModes:,:]
        # Find nTries optimal sets of parameters through conjugate
        # gradient minimization.
        optimal_costs = []
        for k in range(nTries):
            init_theta = self.randInitTheta(0.12)
            w = opt.fmin_cg(self.regCostFunction, 
                            init_theta, 
                            args = (self.Vx, self.Vy, self.lam) )
            optimal_costs.append(
                (self.regCostFunction(w,self.Vx,self.Vy,self.lam),w)
                )
        # Pick the model most closely resembles the data.
        optimalCost, optimalTheta = sorted(optimal_costs)[0]
        
        self.optimalCost = optimalCost
        self.optimalTheta = optimalTheta



def cvLearningCurve():
    """

    Function: CVLEARNINGCURVE() 

    Description -- Uses the dataDynamics class to find an optimal
                   transition matrix for the state space approach
                   being utilized to find the transcriptome
                   dynamics. The function that is used if the file is
                   called directly instead of being imported as a
                   module. Creates cross-validation plots and Learning
                   Curves for the time-series, reduced order dynamics
                   regression problem.

    Parameters -- none, this is really the main() function.

    """
    nCV = 15
    nTest = 5
    data = dataDynamics(
        'data/test/GSE9302_series_matrix.txt',
        nCV,
        nTest,
        nPrune=11
        )
    data.pca()
    data.fit_reduced(3,3,0.0)
    y = np.zeros(data.V[:data.nModes,:].shape)
    y[:,0] = data.V[:data.nModes,0]
    print y.shape, data.optimalTheta.shape
    for k in range(data.V.shape[1]-1):
        y[:,k+1] = np.dot(
            data.optimalTheta.reshape(data.nModes,data.nModes),
            y[:,k]
            )
    pl.plot(y.T)
    pl.axis(hold=True)
    pl.plot(data.V[:data.nModes,:].T)
    pl.show()
    lams = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.]
    errCV = []
    errTrain = []
    for lam in lams:
        data.fit_reduced(3,1, lam)
        cvErr = data.regCostFunction(
            data.optimalTheta, 
            data.Vxcv, 
            data.Vycv, 
            0.0)
        trainErr = data.regCostFunction(
            data.optimalTheta,
            data.Vx,
            data.Vy,
            0.0)
        errCV.append((cvErr,lam))
        errTrain.append(trainErr)

    optLam = sorted(errCV)[0][1]
    errCvPlot = []
    for x in errCV:
        errCvPlot.append(x[0])

    pl.plot(lams,errCvPlot,'g')
    pl.axis(hold=True)
    pl.plot(lams,errTrain,'m')
    pl.show()

    learnCurveCV = []
    learnCurveTrain = []
    mRange = range(data.X.shape[1]-2)
    for k in mRange:
        data.fit_reduced(3,1, optLam,nSamples=k+2)
        trainErr = data.regCostFunction(
                data.optimalTheta,
                data.Vx,
                data.Vy,
                0.0)
        cvErr = data.regCostFunction(
                data.optimalTheta,
                data.Vxcv,
                data.Vycv,
                0.0)
        learnCurveCV.append(cvErr)
        learnCurveTrain.append(trainErr)

    pl.plot(mRange, learnCurveCV,'g')
    pl.axis(hold=True)
    pl.plot(mRange, learnCurveTrain,'m')
    pl.show()
    
if __name__ == '__main__':
    cvLearningCurve()
