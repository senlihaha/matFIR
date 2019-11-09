%% Feature importance ranking (matFIR)
% Shaode Yu
% matFIR: a matlab toolbox for feature importance ranking
% % Email: nicolasyude@163.com    06/07/2018
clear; close all; clc;
warning off

% (1) test 01
addpath( genpath( 'E:\_aFS\aplan\bcdr\' ) );

ben = importfileBCDR( 'abcdr_01_beni.csv' );
benData = table2array( ben( :, 2:18 ) );  benLabel = zeros( size(ben,1), 1 );

mal = importfileBCDR( 'abcdr_01_mali.csv' );
malData = table2array( mal( :, 2:18 ) ); malLabel = ones( size(mal,1), 1 );

X = [ benData; malData ];   Y = [ benLabel; malLabel];

clear ben benData benLabel mal malData malLabel

% % test 02
% % load demo_data
% % 
% % X = data;
% % Y = double(C>2);

%--------------------------------------------------------
% (2) test FIR methods
addpath( genpath( 'auxi' ) );
addpath( genpath( 'firs' ) );

listFIRmethod = { 'fir_asl', 'fir_cor', 'fir_dgufs', 'fir_ec', ...
'fir_fisher', 'fir_fsv', 'fir_gini', 'fir_glsi', ...
'fir_il', 'fir_inf', 'fir_jelsr', 'fir_KruskalWallis', ...
'fir_lapscore', 'fir_llc', 'fir_lnrd', ...
'fir_mat_ttest', 'fir_mat_entropy', ...
'fir_mat_bhattacharyya', 'fir_mat_roc', ...
'fir_mat_wilcoxon', 'fir_mat_relieff', ...
'fir_mat_lasso', 'fir_mc', 'fir_mmls', 'fir_nnsa', ...
'fir_ol', 'fir_pwfp', 'fir_ru', 'fir_sgo', 'fir_soc'};
% 'fir_spider_fisher', 'fir_spider_L0', 'fir_spider_rfe'

rankingFIR = { length(listFIRmethod), 2};
for ii = 1:length(listFIRmethod)
    method = listFIRmethod{ii};
    tmpRanking = ysdFIRcollection( X, Y, method );
    rankingFIR{ii,1} = method;
    rankingFIR{ii,2} = tmpRanking;
end

% 'fisherFS' - no changes
% 'L0FS' - time consuming
% 'refFS' - time consuming 
% ECFS   {-1, +1}
% fsvFS  {-1, +1}




