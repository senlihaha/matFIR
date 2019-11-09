% 'fir_asl', 'fir_cor', 'fir_dgufs', 'fir_ec', ...
% 'fir_fisher', 'fir_fsv', 'fir_gini', 'fir_glsi', ...
% 'fir_il', 'fir_inf', 'fir_jelsr', 'fir_KruskalWallis', ...
% 'fir_lapscore', 'fir_llc', 'fir_lnrd', ...
% 'fir_mat_ttest', 'fir_mat_entropy', ...
% 'fir_mat_bhattacharyya', 'fir_mat_roc', ...
% 'fir_mat_wilcoxon', 'fir_mat_relieff', ...
% 'fir_mat_lasso', 'fir_mc', 'fir_mmls', 'fir_nnsa', ...
% 'fir_ol', 'fir_pwfp', 'fir_ru', 'fir_sgo', 'fir_soc', ...
% 
% 
% 'fir_spider_fisher', 'fir_spider_L0', 'fir_spider_rfe'

%--------------------------------------------------------------------------
%% fs01
% (01) infFS
% 2015. Infinite feature selection

% (02) jelsrFS
% 2011. Feature selection via joint embedding learning and sparse regression

% (03) lapscoreFS
% 2006. Laplacian score for feature selection

% (04) llcFS
% 2010. Feature selection and kernel learning for local learning-based clustering

% (05) mcFS
% 2010. Unsupervised feature selection for multi-cluster data

% (06) nnsaFS
% 2012. Unsupervised feature selection using nonnegative spectral analysis

% (07) rslFS
% 2014. Robust spectral learning for unsupervised feature selection (QUIT)

% (08) ruFS
% 2013. Robust unsupervised feature selection

% (09) socFS
% 2015. Unsupervised simultaneous orthogonal basis clustering feature selection

% (10) sgoFS
% 2016. Unsupervised feature selection with structured graph optimization

% (11) glsiFS
% 2007. Spectral feature selection for supervised and unsupervised learning
%    feature selection based on spectral information of graph laplacian

% (12) lnrdFS
% 2011. L2,1-norm regularized discriminative feature selection for
% unsupervised learning

% (a) GPI
% 2017. A generalized power iteration method for solving quadratic problem on the Stiefel manifold
% Generalized power iteration method (GPI) for solving min_{W'W=I} Tr(W'AW-2W'B)
% -------------------------------------------------------------------------------------

%--------------------------------------------------------------------------
%% fs02
% Quite HARD to follow
% (13) relieffFS
% 1997. Overcoming the myopia of inductive learning algorithms with RELIEFF

% (14) qpFS
% 2010. Quadratic programming feature selection

% (15) pglsmiFS
% 2013. Feature selection via l1-penalized squared-loss mutual information

%--------------------------------------------------------------------------
%% fs03MI
% 2014. Effective global approaches for mutual information based feature selection
% Mutual Informattion (MI) based feature selection

% (01) cifeMIFS
% 2006. Conditional infomax learning: An integrated framework for feature extraction and fusion

% (02) cmimMIFS (Conditional mutual information minimization)
% 2013. Mutual information-based method for selecting informative feature sets.

% (03) maxrelMIFS (maximum relevance)
% 2014. Effective global approaches for mutual information based feature selection
%   Maximum relevance minimum total redundancy  (MRMTR) or extended MRMR (EMRMR) 

% (04) minredMIFS (minimum redundancy)
% 2003. Minimum redundancy feature selection from microarray gene expression data.

% (05) miqMIFS
% 2008. Conditional mutual information based feature selection

% (06) mrmrMIFS
% 2005. Feature selection based on mutual information: 
%               Criteria of max-dependency, max-relevance, and min-redundancy

% (07) mrmtrMIFS
% 2014. Effective global approaches for mutual information based feature selection
%   Maximum relevance minimum total redundancy  (MRMTR) or extended MRMR (EMRMR) 


% (08) qpFS
% 2010. Quadratic programming feature selection

% (09) srgcMIFS (spectral relaxation global conditional mutual information)
% 2014. Effective global approaches for mutual information based feature selection
%   Maximum relevance minimum total redundancy  (MRMTR) or extended MRMR (EMRMR) 

% (10) miFSbattiti
% 1994. Using mutual information for selecting features in supervised neural net learning
%       rank = rank_mifs(X,T,beta) % important features are at last
% another FS methods using  rank = rank_mifsfs(X,T)


%--------------------------------------------------------------------------
%% fs04

% (26) jointMIFS
% 1999. Feature selection based on joint mutual information

% (27) iambFS
% 2003. Algorithms for large scale markov blanket discovery

% (28) mimFS (mutual information maximization)
% 2003. Feature extraction by non-parametric mutual information maximization

% (29) semiIAMB
% (30) semiJMI
% (31) semiMIM
% 2018. Simple strategies for semi-supervised feature selection

% (32) pwfpFS
% 2017. An effective feature selection method based on 
%          pair-wise feature proximity for high dimensional low sample size data

% (33) kfirFS (kernelFIR)
% (34) kmiFS  (kernelMI)
% (35) kplsFS (kernelPLS)
%                   note "Gaussian" or "polynomial" kernel
 % 2014. A kernel-based multivariate feature selection method for microarray data classification
% 
%      case 'fir_kpls_polynomial'
%         fprintf( '===*** 2016. A multi-objective heuristic algorithm for gene expression microarray data classification ...===\n' );
%         X = normalizemeanstd( X_train );
%         Y = binarize( Y_train );
%         
%         num_Component = 10; % number of components
%         alpha = 1;
%         coef = 0.1;
%         
%         Kxx = kernel( X, X, 'polynomial', alpha, coef );
%         Kxy = kernel( X, X([1:2:size(X,1)], : ), 'polynomial', alpha, coef );
%         
%         kplsXS = kernelPLS( Kxx, Kxy, Y, num_Component );
%         
%         kX0 = X - ones( size(X,1), 1 )*mean( X );
%         kWeight = pinv( kX0 )*kplsXS;
%         
%         kVIP = calVIP( Y, kplsXS( :, 1 : num_Component ), kWeight( :, 1 : num_Component ) );
%         
%         [ ~, FeatureRank ] = sort( kVIP, 'descend' );
%         ranking = FeatureRank';
%         
%     case 'fir_kpls_gaussian'
%         fprintf( '===*** 2016. A multi-objective heuristic algorithm for gene expression microarray data classification ...===\n' );
%         X = normalizemeanstd( X_train );
%         Y = binarize( Y_train );
%         
%         num_Component = 10; % number of components
%         
%         Kxx = kernel( X, X, 'gaussian' ); 
%         Kxy = kernel( X, X([1:2:size(X,1)],:), 'gaussian' );
%         
%         kplsXS = kernelPLS( Kxx, Kxy, Y, num_Component );
%         
%         kX0 = X - ones( size(X,1), 1 )*mean( X );
%         kWeight = pinv( kX0 )*kplsXS;
%         
%         kVIP = calVIP( Y, kplsXS( :, 1 : num_Component ), kWeight( :, 1 : num_Component ) );
%         
%         [ ~, FeatureRank ] = sort( kVIP, 'descend' );
%         ranking = FeatureRank';
        
 %--------------------------------------------------------------------------
%% fs05 FSLib2018
% (37) cFS
% 1998. Correlation-based feature subset selection for machine learning
%    pairwise linear correlation coefficient

% (38) dguFS
% 2018. Dependence guided unsupervised feature selection

% (39) ecFS
% 2016. Features selection via eigenvector centrality
% Roffo G, Melzi S. Ranking to Learn[C]//International Workshop on New Frontiers in Mining Complex Patterns. Springer, Cham, 2016: 19-35.

% (40) aslFS
% 2015. Unsupervised feature selection with adaptive structure learning

% (41) fsvFS -- cmfs
% 1998. Feature selection via concave minimization and support vector machines

% (42) 
% 2017. Infinite latent feature selection

% (01) infFS
% 2015. Infinite feature selection

% (03) lapscoreFS
% 2006. Laplacian score for feature selection

% (04) llcFS
% 2010. Feature selection and kernel learning for local learning-based clustering

% (05) mcFS -- mcd-fs
% 2010. Unsupervised feature selection for multi-cluster data

% (16) mrmrMIFS -- mdmrmr-mi-fs
% 2005. Feature selection based on mutual information: 
%               Criteria of max-dependency, max-relevance, and min-redundancy

% (43) mutInfFS (mutual information based feature selection)
% 2014. A review of feature selection methods based on mutual information

% (13) relieffFS
% 1997. Overcoming the myopia of inductive learning algorithms with RELIEFF

% (12) lnrdFS -- UDFS
% 2011. L2,1-norm regularized discriminative feature selection for
% unsupervised learning

% (44) olFS (UFSwithOL)
% 2017. Unsupervised feature selection with ordinal locality

% LASSO
% 2011. Regression shrinkage and selection via the lasso: a retrospective
% (45-49) rkfmatFS
% idx = rankfeatures( X', Y', 'Criterion', a );
%    a = 'ttest', 'entropy', 'bhattacharyya', 'roc', 'wilcoxon'
% Rank key features by class separability criteria
