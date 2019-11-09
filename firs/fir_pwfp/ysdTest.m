addpath( genpath( 'E:\_aFS\bcdr\' ) );

ben = importfileBCDR( 'abcdr_01_beni.csv' );
benData = table2array( ben( :, 2:18 ) );
benLabel = zeros( size(ben,1), 1 );

mal = importfileBCDR( 'abcdr_01_mali.csv' );
malData = table2array( mal( :, 2:18 ) );
malLabel = ones( size(mal,1), 1 );

clear ben mal

% Lesion classification
% data preparation
X = [ benData;  malData ];
Y = [ benLabel; malLabel ];

%---------------------------
% percDim = ;
ranking = pwfp(X,Y, 6)