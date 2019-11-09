function [ranking] = ysdFIRcollection( X_train, Y_train, methodS, numF)
% --------------------------------------------------------------

% --------------------------------------------------------------

% checking input parameters
if nargin<4
    numF = size( X_train, 2 ); % all features
end

if nargin < 3
    fprintf('Wrong: No method is selected ... \n')
    ranking = 0;
    return;
end

labels = unique( Y_train );
if length( labels ) ~= 2
    fprintf( 'Wrong: This FIR code is for binary classification ... \n');
    ranking = 0;
    return;
end

% start feature importance ranking 

selection_method = methodS;

switch (selection_method)    
    % this part comes from FSLib
    %      Feature Selection Library (MATLAB Toolbox)
    case 'fir_asl'
        fprintf( '===*** 2015. Unsupervised feature selection with adaptive structure learning ...===\n' );
        options.lambda1 = 1;
        options.LassoType = 'SLEP';
        options.SLEPrFlag = 1;
        options.SLEPreg = 0.01;
        options.LARSk = 5;
        options.LARSratio = 2;
        nClass = 2;
        W = FSASL(X_train', nClass, options);
        [ ~, ranking ] = sort( abs( W(:,1) ) + abs( W(:,2) ), 'descend' );
        ranking = ranking';
        
    case 'fir_cor'
        fprintf( '===*** 1998. Correlation-based feature subset selection for machine learning ...===\n' );
        ranking = cfs( X_train );
        ranking = ranking';
        
    case 'fir_dgufs'
        fprintf( '===*** 2018. Dependence guided unsupervised feature selection ...===\n' );
        S = dist( X_train' );
        S = -S./max( max(S) ); % it's a similarity
        nClass = 2;
        alpha = 0.5;
        beta = 0.9;
        nSel = 2;
        Y = DGUFS( X_train', nClass, S, alpha, beta, nSel );
        [~,ranking]=sort(Y(:,1)+Y(:,2),'descend');
        ranking = ranking';
          
    case 'fir_ec'
        fprintf( '===*** 2016. Features selection via eigenvector centrality ...===\n' );
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )';
        
    case 'fir_fisher'
        fprintf( '===*** 2011. Generalized fisher score for feature selection ...===\n' );
        ranking = fsFisher( X_train, Y_train );
       
    case 'fir_fsv'
        fprintf( '===*** 2017. Infinite latent feature selection: A probabilistic latent graph-based ranking approach ...===\n' );
        ranking = fsvFS( X_train, Y_train, numF );
        ranking = ranking';
        
    case 'fir_gini'
        fprintf( '===*** 1909. Concentration and dependency ratios ...===\n' );
        ranking = fsGini( X_train, Y_train );    
        
    case 'fir_glsi'
        fprintf( '===*** 2007. Spectral feature selection for supervised and unsupervised learning feature selection based on spectral information of graph laplacian ...===\n' );
        X = X_train';
        num = size( X, 2 );
        k = 5;
        distX = L2_distance_1( X, X );
        [ distX1, idx ] = sort( distX, 2 );
        A = zeros( num );
        for i = 1 : num
            di = distX1( i, 2:k+2 );
            id = idx( i, 2:k+2 );
            A( i, id ) = ( di(k+1) - di )/( k * di(k+1) - sum( di(1:k) ) + eps );
        end
        
        A0 = 0.5 * ( A + A' );
        wFeat = fsSpectrum( A0, X_train, -1);
        [~, ranking] = sort( wFeat, 'descend' );  
        ranking = ranking';

    case 'fir_il'
        fprintf( '===*** 2017. Infinite latent feature selection ...===\n' );
        ranking = ILFS(X_train, Y_train , 6, 0 );
        
    case 'fir_inf'
        fprintf( '===*** 2015. Infinite feature selection ...===\n' );
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        ranking= infFS( X_train , Y_train, alpha , sup , 0 );
      
    case 'fir_jelsr'
        fprintf( '===*** 2011. Feature selection via joint embedding learning and sparse regression ...===\n' );
        data = X_train;
        options.KernelType = 'Gaussian';
        options.t = optSigma(data);
        W_ori = constructKernel(data,[],options);
        
        ReducedDim = 4;        
        alpha = 2;    % 1.5 ~ 2.4       
        beta = 1e-2;  % 1e-2 ~ 1e-1
        
        [W_compute, ~, ~] = jelsr( data, W_ori, ReducedDim, alpha, beta );
        [ ~, ranking ] = sort( sum( W_compute.*W_compute, 2 ), 'descend' );
        ranking = ranking';
        
    case 'fir_KruskalWallis'
        fprintf( '===*** 1973. Nonparametric statistical methods ...===\n' );
        ranking = fsKruskalWallis( X_train, Y_train );
        
    case 'fir_lapscore'
        fprintf( '===*** 2006. Laplacian score for feature selection ...===\n' );
        W = dist( X_train' );
        W = -W./max( W(:) ); % it's a similarity
        lscores = LaplacianScore( X_train, W );
        [~, ranking] = sort(-lscores);
        ranking = ranking';     

    case 'fir_llc'
        fprintf( '===*** 2010. Feature selection and kernel learning for local learning-based clustering ...===\n' );
        ranking = llcfs( X_train );
        ranking = ranking';
                
    case 'fir_lnrd'
        fprintf( '===*** 2011. L2,1-norm regularized discriminative feature selection for unsupervised learning ...===\n' );
        nClass = 2;
        ranking = UDFS( X_train, nClass );
        ranking = ranking';

    % -- matlab embedded algorithm
    case 'fir_mat_ttest'
        fprintf( '===*** 1987. Guinness, Gosset, Fisher, and small samples ...===\n' );
        fprintf( '............ matlab embedded ttest (rankfeatures) ...===\n' );
        ranking = rankfeatures( X_train', Y_train', 'Criterion', 'ttest' );
        ranking = ranking';
        
    case 'fir_mat_entropy'
        fprintf( '===*** 1951. On information and sufficiency ...===\n' );
        fprintf( '............ matlab embedded entropy (rankfeatures) ...===\n' );
        ranking = rankfeatures( X_train', Y_train', 'Criterion', 'entropy' );
        ranking = ranking';
        
    case 'fir_mat_bhattacharyya'
        fprintf( '===*** 1952. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations ...===\n' );
        fprintf( '............ matlab embedded bhattacharyya (rankfeatures) ...===\n' );
        ranking = rankfeatures( X_train', Y_train', 'Criterion', 'bhattacharyya' );
        ranking = ranking';

    case 'fir_mat_roc'
        fprintf( '===*** 1997. The use of the area under the ROC curve in the evaluation of machine learning algorithms ...===\n' );
        fprintf( '............ matlab embedded roc (rankfeatures) ...===\n' );
        ranking = rankfeatures( X_train', Y_train', 'Criterion', 'roc' );
        ranking = ranking';
        
    case 'fir_mat_wilcoxon'
        fprintf( '===*** 1945. Individual comparisons by ranking methods ...===\n' );
        fprintf( '............ matlab embedded wilcoxon (rankfeatures) ...===\n' );
        ranking = rankfeatures( X_train', Y_train', 'Criterion', 'wilcoxon' );
        ranking = ranking';  
        
    case 'fir_mat_relieff'
        fprintf( '===*** 1997. Overcoming the myopia of inductive learning algorithms with RELIEFF ...===\n' );
        fprintf( '............ matlab embedded relieff (reliefF) ...===\n' );
        numNeighbor= 10; % numNeighbor nearest neighbors
        ranking = relieff( X_train, Y_train, numNeighbor);

    case 'fir_mat_lasso'
        fprintf( '===*** 1996. Regression shrinkage and selection via the lasso ...===\n' );
        fprintf( '............ matlab embedded LASSO (lasso) ...===\n' );
        lambda = 25;
        B = lasso( X_train, Y_train );
        [ ~, ranking ] = sort( B( :, lambda ), 'descend' );
        ranking = ranking';   
              
    case 'fir_mc'
        fprintf( '===*** 2010. Unsupervised feature selection for multi-cluster data ...===\n' );
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        FeaIndex = MCFS_p(X_train,numF,options);
        ranking = FeaIndex{1}';  
        
    case 'fir_mmls'
        fprintf( '===*** 2013.Minimum-maximum local structure information for feature selection ...===\n' );
        lamda = 0.5; 
        ranking = fsMMLS( X_train, Y_train, lamda );
        
    case 'fir_nnsa'
        fprintf( '===*** 2012. Unsupervised feature selection using nonnegative spectral analysis ...===\n' );
        X = X_train'; % dim*num
        c = 10;
        num = size(X,2);
        k = 5;
        distX = L2_distance_1(X,X);
        [distX1, idx] = sort(distX,2);
        A = zeros(num);
        for i = 1:num
            di = distX1(i,2:k+2);
            id = idx(i,2:k+2);
            A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
        end
        
        A0 = (A+A')/2;
        
        [Label,L1,~] = sc(A0,c);
        
        tLabel = zeros(num,c);
        for i = 1:num
            tLabel(i,Label(i)) = 1;
        end
        W = zeros(size(X,1),c);
        aa = 0.1;
        bb = 1;
        cc = 1000;
        [~,Wa,~] = AAAI2012(X,full(L1),tLabel,W,100,aa,bb,cc);
        [~, ranking] = sort(sum(Wa.*Wa,2),'descend');
        ranking = ranking';
        
    case 'fir_ol'
        fprintf( '===*** 2017. Unsupervised feature selection with ordinal locality ...===\n' );
        para.p0 = 'sample';
        para.p1 = 1e6;
        para.p2 = 1e2;
        nClass = 2;
        [~,~,ranking,~] = UFSwithOL(X_train',nClass,para);
        ranking = ranking';     
        
    case 'fir_pwfp'
        fprintf( '===*** 2017. An effective feature selection method based on pair-wise feature proximity for high dimensional low sample size data ...===\n' );
        ranking = pwfp( X_train, Y_train );
        
    case 'fir_ru'
        fprintf( '===*** 2013. Robust unsupervised feature selection ...===\n' );
        options.nu = 5;         options.alpha  = 5;
        options.beta = 4;       options.MaxIter = 10;
        options.epsilon = 1e-4; options.verbose = 1;
        
        X = X_train';           c = 26;
        num = size( X, 2 );     k = 5;
        distX = L2_distance_1( X, X );
        [ distX1, idx ] = sort( distX, 2 );
        A = zeros( num );
        for i = 1 : num
            di = distX1( i, 2:k+2 );
            id = idx( i, 2:k+2 );
            A( i, id ) = ( di(k+1) - di )/( k * di(k+1) - sum( di(1:k) ) + eps );
        end
        A0 = 0.5 * ( A + A' );
        [ Label, L1, ~ ] = sc( A0, c );
        
        tLabel = zeros( num, c );
        for i = 1 : num
            tLabel( i, Label(i) ) = 1;
        end
        
        W = RUFS( X_train, L1, tLabel, options );
        
        [ ~, ranking ] = sort( sum( W.*W, 2 ), 'descend' );
        ranking = ranking';
        
    case 'fir_sgo'
        fprintf( '===*** 2016. Unsupervised feature selection with structured graph optimization ...===\n' );
        ranking = SOGFS( X_train', 1000, 10, 15, 5 );
        ranking = ranking';
        
    case 'fir_soc'
        fprintf( '===*** 2015. Unsupervised simultaneous orthogonal basis clustering feature selection ...===\n' );
        data = X_train;
        num = size( data, 2 );
        ranking = SOCFS( data', num, 100,100,50,50 );
        ranking = ranking';
        
    % -- spider-wrapped methods
    case 'fir_spider_fisher'
        fprintf( '===*** 2011. Generalized fisher score for feature selection ...===\n' );
        fprintf( '............ spider-wrapped Fisher score, no changes of feature indexing ...\n');
        selection_method = 'fisher';
        ranking = spider_wrapper( X_train, Y_train, numF, lower(selection_method) );
        
    case 'fir_spider_L0'
        fprintf( '===*** 1995. Sparse approximate solutions to linear systems ...===\n' );
        fprintf( '............ spider-wrapped L0-penalized, time consuming ...\n');
        selection_method = 'l0';
        ranking = spider_wrapper( X_train, Y_train, numF, lower(selection_method) );
        
    case 'fir_spider_rfe'
        fprintf( '===*** 2002. Gene selection for cancer classification using support vector machines ...===\n' );
        fprintf( '............ spider-wrapped SVM-RFE, time consuming ...\n');
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));        
        
    otherwise
        disp('Unknown method.')
end
end

