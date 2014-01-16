#! /usr/bin/env python

import numpy as np
import pylab as pl

from scipy import optimize as opt
np.random.seed(seed=5629387459236459872)

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
    def __init__(self, 
                 fname,
                 nCV, 
                 nTest, 
                 nPrune=0):
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
        self.nGenes = len(self.geneLst)
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
        self.svdFlag = False

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

    def PCA(self):
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
        self.svdFlag = False

    def SVD(self):
        """
        Method: SVD(self)
                     
        Description: performs spectral decomposition of the normalized
        features from the time series data via the SVD.
                                          
        Parameters:  passed through from __init__
                                                                        
        """
        tS = self.tSeries[:,:-(self.nCV+self.nTest)]
        u, s, v = np.linalg.svd(tS)
        self.U = u
        self.S = s
        self.V = v
        self.svdFlag = True
        self.pcaFlag = False

    def randInitTheta(self,
                      init_epsilon):
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

    def regCostFunction(self, 
                        theta, 
                        X, 
                        y, 
                        lam):
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
        nModes =self.nModes
        Theta = theta.reshape((nModes,nModes))
        if self.method == 'pca':
            biasStart = 0
        else:
            biasStart = 1
        return (np.sum((np.dot(Theta,X) - y)**2 )\
                          + lam*np.linalg.norm(Theta[:,biasStart:],ord=1)**2) \
                          /(2*len(X.ravel()))

    def projection(self,
                   method='pca'):
        """
        Method: PROJECTION(self, method) 

        Description: projection() creates the Vxcv, Vxtest, Vycv, and Vytest
                     objects based on whether a PCA or an SVD is applied. Also
                     create Vtrain, Vx, and Vy.

        Parameters: method -- default value is 'pca', other possible value is
                              is 'svd'.
        """
        nSamples = self.nSamples
        nModes = self.nModes
        if method=='pca' and self.pcaFlag==False:
            self.PCA()
        elif method=='svd' and self.svdFlag==False:
            self.SVD()

        uT = self.U.T
        self.Vtrain = self.V[:nModes,self.nPrune:self.nPrune+nSamples]
        if method=='pca':
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

        if method=='svd':
             # Create cv and test sets by rotating by U and scaling by S.  
            self.Vxcv = np.multiply(
                1./self.S[:nModes],
                np.dot(uT[:nModes,:], self.Xcv).T).T
            self.Vxtest = np.multiply(
                1./self.S[:nModes],
                np.dot(uT[:nModes,:],self.Xtest).T).T
            self.Vycv = np.multiply(
                1./self.S[:nModes],
                np.dot(uT[:nModes,:], self.ycv).T).T
            self.Vytest = np.multiply(
                1./self.S[:nModes],
                np.dot(uT[:nModes,:], self.ytest).T).T
        Vx = self.Vtrain[:,:-1]
        Vy = self.Vtrain[:,1:]
        if self.scrambleFlag== False:
            # Scramble the sample features and targets together.
            randV = np.random.permutation(
                np.concatenate((Vx,Vy),axis=0).T
                ).T
            self.Vx = randV[:nModes,:]
            self.Vy = randV[nModes:,:]
            self.scrambleFlag == True
        else:
            self.Vx = Vx
            self.Vy = Vy

    def fit_reduced(self, 
                    nModes, 
                    nTries, 
                    lam, 
                    initEpsilon, 
                    nSamples=None, 
                    method='pca'):
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
        # This is passed through to regCostFunction.
        self.nModes = nModes
        self.lam = lam
        self.method = method
        # If we don't use the nSamples variable, we use the 
        # maximum number of examples in the training set and    
        # get rid of the first nPrune samples.                              
        if nSamples==None:
            self.nSamples = self.X.shape[1]-self.nPrune+1
            self.scrambleFlag = False
        elif nSamples == 1:
            self.nSamples = nSamples
            self.scrambleFlag = False
        elif nSamples > 1:
            self.nSamples = nSamples
            self.scrambleFlag = True
        # Create cv and test sets by rotating by U and scaling by S.
        self.projection(method=method)
        optimal_costs = []
        for k in range(nTries):
            init_theta = self.randInitTheta(initEpsilon)
            #print init_theta.shape
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


    def transition_matrix(self):
        """
        Method: TRANSITION_MATRIX(self) 

        Description: creates a full, microscale transition matrix
                     from the macroscale transition matrix and the
                     topology of the SVD modes
          
        Parameters: passed through from self
                    self.nModes -- degree of macroscale system
                    self.nGenes -- degree of microscale system
                    self.optimalTheta -- model of macroscale system
                    self.U -- topology of the macroscale modes
        """
        nModes = self.nModes
        nGenes = self.nGenes
        P = self.optimalTheta.reshape((nModes,nModes))
        transition = np.zeros((nGenes,nGenes))
        u = self.U
        s = self.S
        # The formula we are encoding is the following:
        # w_k = sum_i( sum_j( s_i * u_ik * T_ji * u_i / s_i ) )   
        for k in range(nGenes):
            for kk in range(nModes):
                wkTemp = np.zeros((nGenes,))
                for kkk in range(nModes):
                    wkTemp += P[kk,kkk] * u[:,kkk] / s[kkk]
                wkTemp *= s[kk]*u[k,kk]
                transition[k,:] += wkTemp
        # Assign the transitionMatrix to the class.
        self.transitionMatrix = transition

    def linSys(self,
               x0, 
               A, 
               tspan):
        """
        Method: LINSYS(x0, A, tspan)
        
        Description: Solved the linear system defined by
                     dy/dt = A * y from the initial conditions x0
                     for times in tspan.
          
        Parameters:  x0 -- initial conditions
                     A -- the constant matrix that also happens
                          to be the Jacobian of the system
                     tspan -- a vector of times to solve the system
        """
        nTime, nDegree = len(tspan), len(x0)
        lam, eta = np.linalg.eig(A)
        c = np.dot(np.linalg.inv(eta), x0)
        mat_lst = [np.multiply(np.exp(lam*t), eta) for t in tspan]
        y = np.dot(mat_lst,c)
        return y


    def linSysCostFunction(self, 
                     Theta, 
                     train,
                     lam):
        """
        Method: LINSYSCOSTFUNCTION(self, Theta, train, lam)

        Descripton: Returns the error between a linear dynamical
                    system with Jacobian Matrix Theta and a data set
                    for training, train.

        Parameters: self -- oop
                    Theta -- the Jacobian of the linear dynamical 
                             system
                    train -- the data we are looking to fit Theta to.
                    lam -- regularization lambda
                   
        """
        x0=train[:,0]
        mRange=range(train.shape[1])
        A = Theta.reshape((train.shape[0],train.shape[0]))
        y=self.linSys(x0,A,mRange)
        return (np.linalg.norm((y-train.T)**2) + 
                              lam*np.linalg.norm(Theta,ord=1)) \
                              /float(len(mRange))

    def dynamic_fit_reduced(self, 
                            nModes, 
                            nTries, 
                            lam, 
                            initEpsilon, 
                            nSamples=None, 
                            method='pca'):
        """

        Method: DYNAMIC_FIT_REDUCED(self, 
                                    nModes, 
                                    nTries, 
                                    lam,
                                    initEpsilon,
                                    nSamples=None,
                                    method='pca')

        Description: A method to fit a linear, discrete-time dynamical
        model to a set of data contained in the class dataDynamics.

        Parameters: nModes -- the degree of the first order system
                              used to model the data
                    nTries -- the number of times to train a model
                    lam -- the regularization parameter
                    initEpsilon -- the amplitude of the random initial 
                                   Theta parameters.
                    nSamples -- the number of samples to include in 
                                the training set.
                    method -- either 'pca' or 'svd'.
       
        """
        # This is passed through to regCostFunction.
        self.nModes = nModes
        self.lam = lam
        # If we don't use the nSamples variable, we use the 
        # maximum number of examples in the training set and    
        # get rid of the first nPrune samples.                                 
        if nSamples==None:
            self.nSamples = self.X.shape[1]-self.nPrune+2
        # We don't want to scramble the data because we are
        # now using a dynamic approach, so we fool the method
        # by telling it the data is already scrambled.
        self.scrambleFlag = True
        # Create cv and test sets by rotating by U and scaling by S.
        self.projection(method=method)
        optimal_costs = []
        for k in range(nTries):
            init_theta = self.randInitTheta(initEpsilon)
            w = opt.fmin_cg(self.linSysCostFunction, 
                            init_theta, 
                            args = (self.Vtrain[:,:nSamples],lam) )
            optimal_costs.append(
                (self.linSysCostFunction(w,self.Vtrain[:,:nSamples],lam),w)
                )
        # Pick the model most closely resembles the data.
        optimalCost, optimalTheta = sorted(optimal_costs)[0]
        
        self.optimalCost = optimalCost
        self.optimalTheta = optimalTheta

def booneValidation(genes, trans_mat, saver=None):
    """
    Function: BOONEVALIDATION(genes,trans_mat)

    Description: Create a smaller interaction matrix from the data
                 published along with the 'Genetic Landscape of a
                 Cell' article from Science.

    Parameters: genes -- a list of genes in S. cerevisiae
                trans_mat -- a transition matrix which predicts dynamics.

    """
    # read in the Boone lab's data
    fh = open('data/test/sgadata_costanzo2009_rawdata_matrix_101120.txt','r')
    data = fh.read().split('\n')[:-1]
    fh.close()

    # turn the list into a list of lists.
    interactions = []
    for x in data:
        interactions.append( x.split('\t') )

    landscape = []
    for i in range(2,len(data)):
        landscape.append(interactions[i][2:])

    landscape = np.array(landscape,dtype=np.float)

    # The Open Reading Frames (ORFs) whose interactions are assayed
    # are the row names. No screening has been done yet.
    orf_rows = []
    for x in interactions[2:]:
        orf_rows.append( x[0] )

    # We want an ORF dictionary so we can map gene name to row number.
    N_orfs = len(orf_rows)
    orf_dict = {}
    for i in range(N_orfs):
        orf_dict[orf_rows[i]] = i

    # The genes in the assay consist of the columns of the first row.
    assay_cols = interactions[0][2:]

    # We want a dictionary so we can index the gene name to column number.
    assay_dict={}
    N_assay = len(assay_cols)
    for i in range(N_assay):
        assay_dict[assay_cols[i]] = i

    # Now we want to only consider those genes that are also referred
    # to in the Li and Klevecz paper, i.e. the genes from the
    # Affymetrix assay.
    screen_assay = []
    for x in assay_cols:
        if x in genes:
            screen_assay.append(x)
    N_screen_cols = len(screen_assay)

    # We do the same thing with the rows of the interaction matrix.
    screen_orfs = []
    for x in orf_rows:
        if x in genes:
            screen_orfs.append(x)
    N_screen_rows = len(screen_orfs)

    trans_dict = {}
    N_genes = len(genes)
    
    for i in range(N_genes):
        trans_dict[genes[i]] = i

    shorter = np.zeros((N_screen_rows, N_assay))
    skinnier = np.zeros((N_screen_rows, N_screen_cols))

    trans_short = np.zeros((N_screen_rows, N_genes))
    trans_skinny = np.zeros((N_screen_rows, N_screen_cols))

    for i in range(N_screen_rows):
        shorter[i,:] = landscape[orf_dict[screen_orfs[i]],:]
        trans_short[i,:] = trans_mat[trans_dict[screen_orfs[i]],:]

    for i in range(N_screen_cols):
        skinnier[:,i] = shorter[:,assay_dict[screen_assay[i]]]
        trans_skinny[:,i] = trans_short[:,trans_dict[screen_assay[i]]]

    if saver != None:
        
        fh = open('screen_col_'+saver+'.pkl','w')
        cPickle.dump(screen_assay, fh)
        fh.close()

        fh = open('screen_row_'+saver+'.pkl','w')
        cPickle.dump(screen_orfs, fh)
        fh.close()

        fh = open('screen_transition_'+saver+'.npy','w')
        np.save(fh, trans_skinny)
        fh.close()

        fh = open('screen_landscape_'+saver+'.npy','w')
        np.save(fh, skinnier)
        fh.close()

    for i in range(skinnier.shape[0]):
        for ii in range(skinnier.shape[1]):
            if np.isnan(skinnier[i,ii]):
                skinnier[i,ii] = 1e4

    return [skinnier, trans_skinny]

def char_fp(mat):
    """
    Function: CHAR_FP(mat)

    Description: Characterizes the fixed point that a square matrix
                 <mat> represents.

    Parameters: mat -- the jacobian matrix of the linear dynamical
                       system

    """
    nabla = np.linalg.det(mat)
    tau = np.trace(mat)
    tol = 10e-6
    state = 'error'
    if nabla < 0.0:
        state = "saddle point"
    elif tau == 0.0:
        state = "non-isolated fixed point"
    elif tau**2 > 4*nabla and tau > 0.0:
        state = "unstable node"
    elif tau**2 == 4*nabla:
        state = "star, degenerate node"
    elif tau**2 < 4*nabla and tau > 0.0:
        state = "unstable spiral"
    elif np.abs(nabla) < tol and nabla > 0.0:
        state = "center"
    elif tau**2 < 4*nabla and tau < 0.0:
        state = "stable spiral"
    elif tau**2 > 4*nabla and tau < 0.0:
        state = "stable node"
    return state


def cvLearningCurve(nModes,method='pca'):
    """

    Function: CVLEARNINGCURVE() 

    Description:   Uses the dataDynamics class to find an optimal
                   transition matrix for the state space approach
                   being utilized to find the transcriptome
                   dynamics. The function that is used if the file is
                   called directly instead of being imported as a
                   module. Creates cross-validation plots and Learning
                   Curves for the time-series, reduced order dynamics
                   regression problem.

    Parameters: nModes -- The number of modes to use when modeling the
                          system.
                method -- Either 'pca' or 'svd'. Just says whether to 
                          normalize the features before doing svd.

    """
    nCV = 10
    nTest = 10
    data = dataDynamics(
        'data/test/GSE9302_series_matrix.txt',
        nCV,
        nTest,
        nPrune=11
        )

    ###!!!-------------------------------------------
    # Start a list of values that is to be returned
    # when the function is called.
    ###!!!-------------------------------------------
    whatIwant = []

    # A good bit of coupling between the modes works well.
    initEpsilon = 0.2
    # Do multivariate linear regression with l1 regularization
    lams = np.linspace(0.,1.0,20)
    errCV = []
    errTrain = []
    for lam in lams:
        data.fit_reduced(nModes,1, lam, initEpsilon,method=method)
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
        errCV.append((cvErr,lam,data.optimalTheta))
        errTrain.append(trainErr)

    optLam = sorted(errCV)[0][1]
    theta = sorted(errCV)[0][2]

    Vtest = np.zeros((data.Vxtest.shape[0],nTest+1))
    Vtest[:,:-1] = data.Vxtest
    Vtest[:,-1] = data.Vytest[:,-1]

    y = np.zeros(Vtest.shape)                           
    y[:,0] = Vtest[:,0]

    for k in range(nTest-1):
        y[:,k+1] = np.dot(
            theta.reshape((data.nModes,data.nModes)),
            y[:,k]
            )
    ###!!!-------------------------------------------
    # Return the optimal lambda, errCvPlot, errTrain,
    # and optimal theta parameters.
    ###!!!-------------------------------------------
    whatIwant.append(theta)
    whatIwant.append(y)
    whatIwant.append(data.V)
    whatIwant.append(optLam)
    whatIwant.append(errTrain)
    whatIwant.append(errCV)
    print "The Optimal Lambda was "+str(optLam)+"."
    learnCurveCV = []
    learnCurveTrain = []
    mRange = range(data.X.shape[1]-(1+data.nPrune))
    for k in mRange:
        data.fit_reduced(nModes,1, optLam,initEpsilon, nSamples=k+2,method=method)
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
    whatIwant.append(learnCurveCV)
    whatIwant.append(learnCurveTrain)
    whatIwant.append(data.S)
    whatIwant.append(lams)

    transMat = data.transitionMatrix
    whatIwant.append(transMat)
    whatIwant.append(data.U)

    Vcv = np.zeros((data.Vxcv.shape[0],nCV+1))
    Vcv[:,:-1] = data.Vxcv
    Vcv[:,-1] = data.Vycv[:,-1]

    whatIwant.append(Vcv)
    whatIwant.append(Vtest)

    if method=='pca':
        whatIwant.append(data.mu)
        whatIwant.append(data.sigma)
    else:
        whatIwant.append(np.zeros((len(data.geneLst),)))
        whatIwant.append(np.ones((len(data.geneLst),)))

    whatIwant.append(data.tSeries)
    whatIwant.append(data.geneLst)
    x = whatIwant
    retDict = {'optimalTheta':x[0], 'y':x[1], 'v':x[2],
               'optLam':x[3], 'errTrain':x[4], 'errCV':x[5],
               'learnCurveCV':x[6], 'learnCurveTrain':x[7],
               's':x[8], 'lambdas':x[9],'transMat':x[10],
               'u':x[11], 'Vcv':x[12], 'Vtest':x[13], 
               'mu':x[14], 'sigma':x[15], 'tSeries':x[16],
               'genes':x[17]}
    
    return retDict

def dynamicCvLearningCurve(nModes,method='pca'):
    """
    Function: DYNAMICCVLEARNINGCURVE(nModes,method='pca')
    
    Description:   Uses the dataDynamics class to find an optimal  
                   transition matrix for the continuous approach        
                   being utilized to find the transcriptome                    
                   dynamics. The function that is used if the file is  
                   called directly instead of being imported as a 
                   module. Creates cross-validation plots and Learning 
                   Curves for the time-series, reduced order dynamics 
                   initial value problem. 

    Parameters: nModes -- The number of modes to use when modeling the 
                          system. 
                method -- Either 'pca' or 'svd'. Just says whether to  
                          normalize the features before doing svd. 
    """
    nCV = 10
    nTest = 10
    data = dataDynamics(
        'data/test/GSE9302_series_matrix.txt',
        nCV,
        nTest,
        nPrune=11
        )

    whatIwant = []
    initEpsilon = 0.2
    lams = np.linspace(0.,1.0,20)
    data.dynamic_fit_reduced(nModes,1, 0.0, initEpsilon,method=method)
    Vcv = np.zeros((data.Vxcv.shape[0],nCV+1))
    Vcv[:,:-1] = data.Vxcv
    Vcv[:,-1] = data.Vycv[:,-1]

    errCV = []
    errTrain = []
    for lam in lams:
        data.dynamic_fit_reduced(nModes,1, lam, initEpsilon,method=method)
        cvErr = data.linSysCostFunction(
            data.optimalTheta,
            Vcv,
            0.0)
        trainErr = data.linSysCostFunction(
            data.optimalTheta,
            data.Vtrain,
            0.0)
        errCV.append((cvErr,lam,data.optimalTheta))
        errTrain.append(trainErr)


    optLam = sorted(errCV)[0][1]
    print "The Optimal Lambda was "+str(optLam)+"."

    theta = sorted(errCV)[0][2]
    train = data.Vtrain

    Vtest = np.zeros((data.Vxtest.shape[0],nTest+1))
    Vtest[:,:-1] = data.Vxtest
    Vtest[:,-1] = data.Vytest[:,-1]

    x0 = Vtest[:,0]
    y = data.linSys(x0,
                    theta.reshape((nModes,nModes)),
                    range(Vtest.shape[1]))


    whatIwant.append(theta)
    whatIwant.append(y)
    whatIwant.append(data.Vtrain)
    whatIwant.append(optLam)
    whatIwant.append(errTrain)
    whatIwant.append(errCV)


    learnCurveCV = []
    learnCurveTrain = []
    mRange = range(data.X.shape[1]-(data.nPrune))
    for k in mRange:
        nSamples=k+2
        data.dynamic_fit_reduced(nModes,1, optLam,initEpsilon,nSamples=nSamples,method=method)
        trainErr = data.linSysCostFunction(
                data.optimalTheta,
                data.Vtrain[:,:nSamples],
                0.0)
        cvErr = data.linSysCostFunction(
                data.optimalTheta,
                Vcv,
                0.0)
        learnCurveCV.append(cvErr)
        learnCurveTrain.append(trainErr)


    whatIwant.append(learnCurveCV)
    whatIwant.append(learnCurveTrain)
    whatIwant.append(data.S)
    whatIwant.append(lams)
    whatIwant.append(data.U)

    whatIwant.append(Vcv)
    whatIwant.append(Vtest)
    whatIwant.append(data.tSeries)
    whatIwant.append(data.geneLst)
    if method=='pca':
        whatIwant.append(data.mu)
        whatIwant.append(data.sigma)
    else:
        whatIwant.append(np.zeros((len(data.geneLst),)))
        whatIwant.append(np.ones((len(data.geneLst),)))

    transMat = data.transition_matrix()
    whatIwant.append(transMat)
    x = whatIwant

    retDict = {'optimalTheta':x[0], 'y':x[1], 'v':x[2],
               'optLam':x[3], 'errTrain':x[4], 'errCV':x[5],
               'learnCurveCV':x[6], 'learnCurveTrain':x[7],
               's':x[8], 'lambdas':x[9], 'u':x[10],
               'Vcv':x[11], 'Vtest':x[12], 'tSeries':x[13],
               'genes':x[14], 'mu':x[15], 'sigma':x[16],
               'transMat':x[17]}


    return retDict

def sayHello():
    print "Hello!"

def showCumVar(s1,
               s2,
               labS1='Normalized Features',
               labS2='Non-Normalized Features',
               xLab='Mode',
               yLab='Cumulative Variance',
               title='Cumulative Variance of SVD modes',
               locProp=(4,{'size':14})):

    """
    Function: SHOWCUMVAR(s1,s2,labS1,labS2,xlab,ylab,title,locProp)

    Description: Plot the Cumulative Variance of normalized and non-normalized
    SVD modes.

    Parameters: s1 -- first values of s matrix from an SVD decomposition
                s2 -- second s vector from SVD decomposition
                labS1 -- legend label for s1
                labS2 -- legend label for s2
                xLab -- label for x-axis on plot
                yLab -- label for y-axis on plot
                title -- title of plot
                locProp -- (location,fontsize) tuple for legend() 

    """

    import pylab as pl
    cumVar1norm = sum(s1**2)
    cumVar2norm = sum(s2**2)
    cumVar1, cumVar2 = [], []
    for k in range(len(s1)):
        cumVar1.append(sum(s1[:k]**2))
        cumVar2.append(sum(s2[:k]**2))
    cumVar1 /= cumVar1norm
    cumVar2 /= cumVar2norm
    pl.plot(cumVar1,label=labS1)
    pl.axis(hold=True)
    pl.plot(cumVar2,label=labS2)
    pl.xlabel(xLab)
    pl.ylabel(yLab)
    pl.title(title)
    pl.legend(loc=locProp[0],prop=locProp[1])

    pl.show()


def showCVCurves(errTrains,
                 errCVs,
                 lams,
                 cvLab='CV Error',
                 trainLab='Training Error',
                 xLab=r'Regularization $\lambda$',
                 yLab='Error',
                 titles=('Normalized Discrete',
                         'Non-Normalized Discrete',
                         'Normalized Continuous',
                         'Non-Normalized Continuous'),
                 supTitle=
                 r'Cross Validation of Regularization Parameter $\lambda$',
                 prop={'size':8}
                 ):
    """
    Function: SHOWCVCURVES(cvTrains,
                           cvErrs,
                           lams,
                           cvLab,
                           trainLab,
                           xLab,
                           yLab,
                           titles,
                           supTitle)

    Description: Plot the result of using the optimalTheta matrices to
                 predict the macroscale dynamics.

    Parameters: cvTrains -- list of Cross-Validation training errors
                cvErrs -- list of errors on CV set
                lams -- regularization lambdas
                cvLab -- labels for cvErr
                trainLab -- labels for cvTrain
                xLab -- xlabel
                yLab -- ylabel
                titles -- list of titles for subplots
                supTitle -- SuperTitle!

    """

    for k in range(4):
        pl.subplot(2,2,k+1)
        pl.plot(lams,errTrains[k],label=trainLab)
        pl.axis(hold=True)
        pl.plot(lams,errCVs[k],label=cvLab)
        pl.xlabel(xLab)
        pl.ylabel(yLab)
        pl.title(titles[k])
        pl.legend(prop=prop)
    pl.suptitle(supTitle)

    pl.show()


def showLearningCurves(errTrains,
                       errCVs,
                       lams,
                       cvLab='CV Error',
                       trainLab='Training Error',
                       xLab='Number of Samples',
                       yLab='Error',
                       titles=('Normalized Discrete',
                               'Non-Normalized Discrete',
                               'Normalized Continuous',
                               'Non-Normalized Continuous'),
                       supTitle='Learning Curves',
                       prop={'size':8}
                       ):
    """
    Function: SHOWLEARNINGCURVES(errTrains,
                                 errCVs,
                                 lams,
                                 cvLab,
                                 trainLab,
                                 xLab,
                                 yLab,
                                 titles,
                                 supTitle)

    Description: Plot the Learning Curves associated with the four
                 methods.

    Parameters: see __doc__ for SHOWLEARNINGCURVES()

    """
    import pylab as pl
    for k in range(4):
        pl.subplot(2,2,k+1)
        pl.plot(lams[k],errTrains[k],label=trainLab)
        pl.axis(hold=True)
        pl.plot(lams[k],errCVs[k],label=cvLab)
        pl.xlabel(xLab)
        pl.ylabel(yLab)
        pl.title(titles[k])
        pl.legend(prop=prop)
    pl.suptitle(supTitle)

    pl.show()

def prepResults(results):
    """
    Function: PREPRESULTS(results)

    Description: Prepare results for plotting.

    Parameters: results -- the results from gatherResults()

    """
    models = []
    models.append(results[0]['y'][0,:])
    models.append(results[1]['y'][1,:])
    models.append(results[0]['y'][1,:])
    models.append(results[1]['y'][2,:])
    models.append(results[0]['y'][2,:])
    models.append(results[1]['y'][3,:])
    models.append(results[2]['y'][:,0])
    models.append(results[3]['y'][:,1])
    models.append(results[2]['y'][:,1])
    models.append(results[3]['y'][:,2])
    models.append(results[2]['y'][:,2])
    models.append(results[3]['y'][:,3])

    datas = []
    datas.append(results[0]['Vtest'][0,:])
    datas.append(results[1]['Vtest'][1,:])
    datas.append(results[0]['Vtest'][1,:])
    datas.append(results[1]['Vtest'][2,:])
    datas.append(results[0]['Vtest'][2,:])
    datas.append(results[1]['Vtest'][3,:])
    datas.append(results[2]['Vtest'][0,:])
    datas.append(results[3]['Vtest'][1,:])
    datas.append(results[2]['Vtest'][1,:])
    datas.append(results[3]['Vtest'][2,:])
    datas.append(results[2]['Vtest'][2,:])
    datas.append(results[3]['Vtest'][3,:])

    
    nTest = len(datas[-1])

    labels= []
    labels.append('Mode 1')
    labels.append('Mode 1')
    labels.append('Mode 2')
    labels.append('Mode 2')
    labels.append('Mode 3')
    labels.append('Mode 3')

    labels = 2*labels

    t_begin, t_final = 13.833, 16.9671071428572
    time = np.linspace(t_begin,t_final,48)

    mu = results[0]['mu']
    sigma = results[0]['sigma']

    SIGMA = np.tile(sigma,(nTest,1)).T
    MU = np.tile(mu,(nTest,1)).T


    tSeries = results[0]['tSeries']
    testSet = tSeries[:,-nTest:]
    zsTest = np.multiply(testSet-MU,1./SIGMA)

    pred1 = np.zeros(testSet.shape)
    pred1[:,0] = zsTest[:,0]
    for k in range(nTest-1):
        pred1[:,k+1] = np.dot(results[0]['transMat'],pred1[:,k])

    pred2 = np.zeros(testSet.shape)
    pred2[:,0] = testSet[:,0]
    for k in range(nTest-1):
        pred2[:,k+1] = np.dot(results[1]['transMat'],pred2[:,k])

    pred3 = np.zeros(testSet.shape)

    for k in range(nTest):
        for kk in range(3):
            pred3[:,k] += results[2]['y'].T[kk,k]*results[2]['u'][:,kk]*results[2]['s'][kk]

    pred4 = np.zeros(testSet.shape)

    for k in range(nTest):
        for kk in range(4):
            pred4[:,k] += results[3]['y'].T[kk,k]*results[3]['u'][:,kk]*results[3]['s'][kk]


    preds = [np.multiply(SIGMA,pred1)+MU,
             pred2,
             np.multiply(SIGMA,pred3)+MU,
             pred4]

    zPreds = [pred1,
              np.multiply(pred2-MU,1./SIGMA),
              pred3,
              np.multiply(pred4-MU,1./SIGMA)]


    return [models, datas, labels, time, preds, zPreds]

def filterAssociations(results):
    """
    Function: FILTERASSOCIATIONS(results)

    Description: filters out junk from transition matrices so we can
                 examine the correlation between physical genome-wide
                 associations and collective behavior between modules
                 of genes.

    Parameters: results -- results from gatherResults()

    """

    normTrans = results[5]
    nonNormTrans = results[7]
    landscape = results[6]
    LAND = np.abs(1./landscape) < 0.01
    NORMED = np.abs(normTrans) > 0.0005
    NONNORMED =np.abs(nonNormTrans) > 0.001

    return [LAND, NORMED, NONNORMED]

def showTestDynamics(models,
                     datas,
                     labels,
                     time,
                     xLab='Time',
                     yLab='Expression',
                     locProp=(2,{'size':7}),
                     titles=('Normalized Discrete',
                             'Non-Normalized Discrete',
                             'Normalized Continuous',
                             'Non-Normalized Continuous'),
                     supTitle='Macroscale Data and Dynamic Models'):

    """
    Function: SHOWTESTDYNAMICS(models,
                               datas,
                               labels,
                               time,
                               xLab,
                               yLab,
                               locProp,
                               titles,
                               supTitle)

    Description: Plots the dynamics of the test IVP using either a
                 machine learning/regression approach or an approach 
                 that fits an optimal LDS as an IVP.

    Parameters: models -- list of models to plot
                datas -- list of data to plot
                labels -- labels for the data                  
                time -- times of measurement
                xLab -- xlabel
                yLab -- ylabel
                locProp -- the location and prop for legend()
                titles -- titles of subplots
                supTitle -- SuperiorTitle!

    """
    import pylab as pl
    import numpy as np

    for k in range(12):
        pl.subplot(6,2,k+1)
        pl.plot(time,datas[k],label=labels[k])
        pl.axis(hold=True)
        corr = np.corrcoef(models[k],datas[k])[0,1]
        pl.plot(time,models[k],label='Correlation '+str(corr)[:6])
        pl.legend(loc=locProp[0],prop=locProp[1])
        if k == 10 or k == 11:
            pl.xlabel(xLab)
        else:
            pl.xticks([])
        if k == 0:
            pl.title(titles[0])
        elif k == 1:
            pl.title(titles[1])
        elif k == 6:
            pl.title(titles[2])
        elif k == 7:
            pl.title(titles[3])
        pl.ylabel(yLab)
    pl.suptitle(supTitle)
    pl.show()

def showMicroSurface(preds,
                     zPreds,
                     tSeries,
                     time,
                     yLab='Gene',
                     xLab='Time',
                     zLab='Expression',
                     titles=('Normalized Discrete',
                             'Non-Normalized Discrete',
                             'Normalized Continuous',
                             'Non-Normalized Continuous'),
                     supTitle='Predictions of Microscale Dynamics'
                     ):
    """
    """

    from mpl_toolkits.mplot3d import Axes3D

    nGenes, nTest = preds[0].shape
    y = range(nGenes)
    T, Y = np.meshgrid(time,y)
    testSet = tSeries[:,-nTest:]
    corrs = []
    for k in range(4):
        corrs.append(str(np.corrcoef(preds[k].ravel(),
                                     testSet.ravel())[0,1])[:5])
    fig = pl.figure()
    for k in range(4):
        ax = fig.add_subplot(2,2,k+1,projection='3d')
        surf = ax.plot_surface(T,Y,zPreds[k],cmap=pl.cm.coolwarm)
        ax.set_yticks([])
        ax.set_ylabel(yLab)
        ax.set_xlabel(xLab)
        ax.set_zlabel(zLab)
        ax.view_init(elev=45., azim=315.)
        pl.title(titles[k])
        fig.colorbar(surf,shrink=0.5,aspect=10)
        txt = ax.text(16.7,4000,zPreds[k].max()+1,r'$R^2$='+corrs[k])
    pl.suptitle(supTitle)
    pl.show()

def spyLandTrans(land,norm,nonNorm):
    pl.subplot(3,1,1)
    pl.spy(land,marker='.',markersize=0.1,color='b')
    pl.ylabel('Assayed Genes',{'fontsize':12})
    pl.yticks([])
    pl.xticks([])
    pl.title('Genetic Landscape')
    pl.subplot(3,1,2)
    pl.spy(norm,marker='.',markersize=0.1,color='g')
    pl.yticks([])
    pl.xticks([])
    pl.ylabel('Assayed Genes',{'fontsize':12})
    pl.title('Transition Matrix - Normalized Features')
    pl.subplot(3,1,3)
    pl.spy(nonNorm,marker='.',markersize=0.1,color='m')
    pl.ylabel('Assayed Genes',{'fontsize':12})
    pl.xlabel('Screened Genes',{'fontsize':12})
    pl.xticks([])
    pl.yticks([])
    pl.title('Transition Matrix - Non-Normalized Features')
    pl.suptitle('Comparison of Transition Matrices to Genetic Landscape')

    pl.show()


def gatherResults():
    """
    Function: GATHERRESULTS()
    """
    pd = cvLearningCurve(3,method='pca')
    sd = cvLearningCurve(4,method='svd')
    pc = dynamicCvLearningCurve(3,method='pca')
    sc = dynamicCvLearningCurve(4,method='svd')
    pd_gene, pd_mod = booneValidation(pd['genes'],pd['transMat'])
    sd_gene, sd_mod = booneValidation(sd['genes'],sd['transMat'])
    
    return [pd, sd, pc, sc, pd_gene, pd_mod, sd_gene, sd_mod]

if __name__ == '__main__':
    results = gatherResults()
