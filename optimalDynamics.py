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
        # nModes is pulled out from the class.
        # "This is the beauty of OOP. I can pull variables in
        #  from the class when I know a method is only going
        #  to be called by another method that assigns 
        #  self.nModes with the correct value." -- observation
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
            #print "scrambling"
            # Scramble the sample features and targets together.
            randV = np.random.permutation(
                np.concatenate((Vx,Vy),axis=0).T
                ).T
            self.Vx = randV[:nModes,:]
            self.Vy = randV[nModes:,:]
            self.scrambleFlag == True
        else:
            #print "pre-scrambled"
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
    """Create a smaller interaction matrix from the data published
    along with the 'Genetic Landscape of a Cell' article from
    Science."""

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
    #print N_orfs

    orf_dict = {}
    for i in range(N_orfs):
        orf_dict[orf_rows[i]] = i

    # The genes in the assay consist of the columns of the first row.
    assay_cols = interactions[0][2:]

    # We want a dictionary so we can index the gene name to column number.
    assay_dict={}
    N_assay = len(assay_cols)
    #print N_assay

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

    #print "These are the shapes of the screened transition and landscape matrices"
    #print trans_skinny.shape, skinnier.shape
    #print "This is the shape of trans_mat..."
    #print trans_mat.shape
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


    
    #correlation = np.corrcoef(skinnier.ravel(),1./trans_skinny.ravel())

    #print( "The correlation between the functional and physical transcriptome topologies is %s." % correlation[0,1] )

    #transitionCutoff = np.abs(1./trans_skinny) < 0.05
    #landscapeCutoff = np.abs(skinnier) < 0.05


    return [trans_skinny, skinnier]



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
    # Start a list of values that are to be returned
    # when the function is called.
    ###!!!-------------------------------------------
    whatIwant = []

    #pl.plot(1.-data.S**2/sum(data.S**2))
    #pl.show()

    # whatIwant.append(data.S)
    # A good bit of coupling between the modes works well.
    initEpsilon = 0.2
    # Do multivariate linear regression with l1 regularization
    # data.fit_reduced(nModes,3,0.0,initEpsilon,method=method)
    # Set up Initial Conditions and a place to store the
    # dynamics of the discrete-time linear system.
    #y = np.zeros(data.V[:data.nModes,:].shape)
    #y[:,0] = data.V[:data.nModes,0]

    ###!!!-------------------------------------------
    # Return y and data.V instead of plotting inside
    # the function.
    ###!!!-------------------------------------------
    #whatIwant.append(y)
    #whatIwant.append(data.V)
    # This happens after the cross-validation.

    """
    for k in range(data.nModes):
        pl.subplot(data.nModes,1,k+1)
        pl.plot(y[k,:].T)
        pl.axis(hold=True)
        pl.plot(data.V[k,:].T)
    pl.show()
    """
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
        errCV.append((cvErr,lam))
        errTrain.append(trainErr)

    optLam = sorted(errCV)[0][1]

    y = np.zeros(data.V[:data.nModes,:].shape)                                                                                
    y[:,0] = data.V[:data.nModes,0]

    for k in range(data.V.shape[1]-1):
        y[:,k+1] = np.dot(
            data.optimalTheta.reshape(data.nModes,data.nModes),
            y[:,k]
            )


    ###!!!-------------------------------------------
    # Return the optimal lambda, errCvPlot, errTrain,
    # and optimal theta parameters.
    ###!!!-------------------------------------------
    whatIwant.append(data.optimalTheta)
    whatIwant.append(y)
    whatIwant.append(data.V)
    whatIwant.append(optLam)
    whatIwant.append(errTrain)
    whatIwant.append(errCV)
    # whatIwant = [optimalTheta, y, v, optLam, errTrain, errCV]
    print "The Optimal Lambda was "+str(optLam)+"."
    """
    errCvPlot = []
    
    for x in errCV:
        errCvPlot.append(x[0])
    
    pl.plot(lams,errCvPlot,'g')
    pl.axis(hold=True)
    pl.plot(lams,errTrain,'m')
    pl.show()
    """
    ###!!!-------------------------------------------
    # Return learnCurveCV and learnCurveTrain instead
    # of plotting inside the function
    ###!!!-------------------------------------------
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
    # whatIwant = [optimalTheta, y, v, optLam, errTrain, errCV,
    #              learnCurveCV, learnCurveTrain, S, lams]

    data.transition_matrix()
    transMat = data.transitionMatrix
    whatIwant.append(transMat)
    whatIwant.append(data.U)

    Vcv = np.zeros((data.Vxcv.shape[0],nCV+1))
    Vcv[:,:-1] = data.Vxcv
    Vcv[:,-1] = data.Vycv[:,-1]

    Vtest = np.zeros((data.Vxtest.shape[0],nTest+1))
    Vtest[:,:-1] = data.Vxtest
    Vtest[:,-1] = data.Vytest[:,-1]

    whatIwant.append(Vcv)
    whatIwant.append(Vtest)
    whatIwant.append(data.mu)
    whatIwant.append(data.sigma)
    whatIwant.append(data.tSeries)

    x = whatIwant
    retDict = {'optimalTheta':x[0], 'y':x[1], 'v':x[2],
               'optLam':x[3], 'errTrain':x[4], 'errCV':x[5],
               'learnCurveCV':x[6], 'learnCurveTrain':x[7],
               's':x[8], 'lambdas':x[9],'transMat':x[10],
               'u':x[11], 'Vcv':x[12], 'Vtest':x[13], 
               'mu':x[14], 'sigma':x[15], 'tSeries':x[16]}
    
    """
    pl.plot(mRange, learnCurveCV,'g')
    pl.axis(hold=True)
    pl.plot(mRange, learnCurveTrain,'m')
    pl.show()
    """
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
    ###!!!-------------------------------------------
    # So return data.S instead of plotting inside the 
    # function.
    ###!!!-------------------------------------------
    whatIwant = []
    initEpsilon = 0.2
    lams = np.linspace(0.,1.0,20)
    errCV = []
    errTrain = []
    for lam in lams:
        data.dynamic_fit_reduced(nModes,1, lam, initEpsilon,method=method)
        cvErr = data.linSysCostFunction(
            data.optimalTheta,
            data.Vtrain,
            0.0)
        trainErr = data.linSysCostFunction(
            data.optimalTheta,
            data.Vtrain,
            0.0)
        errCV.append((cvErr,lam))
        errTrain.append(trainErr)


    optLam = sorted(errCV)[0][1]

    theta = data.optimalTheta.reshape((nModes,nModes))
    train = data.Vtrain
    x0 = data.Vtrain[:,0]
    y = data.linSys(x0,theta,range(data.Vtrain.shape[1]))


    whatIwant.append(data.optimalTheta)
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
                data.Vtrain[:,:nSamples],
                0.0)
        learnCurveCV.append(cvErr)
        learnCurveTrain.append(trainErr)


    whatIwant.append(learnCurveCV)
    whatIwant.append(learnCurveTrain)
    whatIwant.append(data.S)
    whatIwant.append(lams)
    whatIwant.append(data.U)

    Vcv = np.zeros((data.Vxcv.shape[0],nCV+1))
    Vcv[:,:-1] = data.Vxcv
    Vcv[:,-1] = data.Vycv[:,-1]

    Vtest = np.zeros((data.Vxtest.shape[0],nTest+1))
    Vtest[:,:-1] = data.Vxtest
    Vtest[:,-1] = data.Vytest[:,-1]

    whatIwant.append(Vcv)
    whatIwant.append(Vtest)
    whatIwant.append(data.mu)
    whatIwant.append(data.sigma)
    whatIwant.append(data.tSeries)

    x = whatIwant

    retDict = {'optimalTheta':x[0], 'y':x[1], 'v':x[2],
               'optLam':x[3], 'errTrain':x[4], 'errCV':x[5],
               'learnCurveCV':x[6], 'learnCurveTrain':x[7],
               's':x[8], 'lambdas':x[9], 'u':x[10],
               'Vcv':x[11], 'Vtest':x[12], 'mu':x[13],
               'sigma':x[14], 'tSeries':x[15]}


    return retDict


    ###!!!-------------------------------------------
    # Return y and data.V instead of plotting inside
    # the function.
    ###!!!-------------------------------------------
    """
    for k in range(data.nModes):
        pl.subplot(data.nModes,1,k+1)
        pl.plot(y[k,:].T)
        pl.axis(hold=True)
        pl.plot(data.V[k,:].T)
    pl.show()

    pl.plot(y),pl.show()
    pl.plot(y.T),pl.show()
    """


def primary():
    ###!!!-------------------------------------------
    # This is going to be where the results are 
    # gathered and plotted. This will be fun.
    ###!!!-------------------------------------------
    
    # All are commented out for the time being.
    pcaDiscrete = cvLearningCurve(3,method='pca')
    svdDiscrete = cvLearningCurve(4,method='svd')
    pcaContinuous = dynamicCvLearningCurve(3,method='pca')
    svdContinuous = dynamicCvLearningCurve(4,method='svd')
    

if __name__ == '__main__':
    primary()



"""
def plot4(tspan, h, train, test, lam, fname=None):
    """"""
    PLOT4(tspan, h, train, test) -- tspan is the time domain, 
                                 h is the model, 
                                 y is the training data, 
                                 test is the test set.

    Description: Make a 2 X 2 subplot of the four modes. Make the
    predictions one color (uniform for each 4 modes) and the data
    another color. Make the test set a different color from the
    training data. Label with individual correlations using a legend.
    Calculate the correlations inside plot4.
    """"""
    nTest = test.shape[1]
    rSquares = []
    for k in range(4):
        rSquares.append(np.corrcoef(h[k,-nTest:],test[k,:])[0,1])

    intStrings = [ '1','2','3','4']
    modelLabels = ['Dynamical Model '+x for x in intStrings]
    dataLabels = ['Genome Mode '+x for x in intStrings]
    testLabels = ['Model-Test Correlation= %.4f' % x for x in rSquares ]

    rSquares = []
    for k in range(4):
        rSquares.append(np.corrcoef(h[k,-nTest:],test[k,:])[0,1])
    
    testPlot = np.zeros((test.shape[0],test.shape[1]+1))
    testPlot[:,0] = train[:,-1]
    testPlot[:,1:] = test
    pl.figure()
    pl.suptitle('Models and Data for Regularization Lambda = %.2f' % lam)
    for k in range(4):
        pl.subplot(2,2,k+1)
        pl.plot(tspan[:-nTest],
                train[k,:],
                'g.-',
                label=dataLabels[k],
                markersize=6)
        if k==0 or k==2:
            pl.xlabel('time',fontsize=6)
            pl.ylabel('expression', fontsize=6)
        else:
            pl.xlabel('time',fontsize=6)
            pl.ylabel('')

        pl.axis(hold=True)
        pl.plot(tspan[-nTest-1:],
                testPlot[k,:],
                'm.-',
                label=testLabels[k],
                markersize=6)

        pl.plot(tspan, 
                h[k,:],
                'y^-',
                label=modelLabels[k],
                markersize=6)
        if k == 0:
            location = 3
        else:
            location = 2
        pl.legend(loc=location ,prop={'size':6})
        if fname == None:
            pl.show()
        else:
            pl.savefig(fname)
    pl.clf()

"""
