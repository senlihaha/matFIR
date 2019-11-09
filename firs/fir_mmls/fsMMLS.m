function ranking = fsMMLS( data, label, lamda, k )
% ---------------------------------------------------
% fsMMLS
% Please cite this paper:
%   Minimum-maximum local structure information for feature selection
% ---------------------------------------------------
% Input:
%      data     m*n, m sample and n feature per sample
%      label    m*1, m sample and 1 label per sample
%      lamda    ?
%         k     k-nn
%  Output:
%      ranking  1*n, 1 row with indexing for features
% ---------------------------------------------------

if nargin < 4
    k = 5;
end
options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = k;


Wt = constructW(data, options);

[nsmp, nfea] = size(data);
Ww = full(Wt);
for i = 1:nsmp
    for j = 1:nsmp
        if label(i) ~= label(j)
            Ww(i, j) = 0;
        end
    end
end

Wb = Wt - Ww;
A = lamda*Ww - (1-lamda)*Wb;
D = diag(sum(A, 2));
L = D - A;
Dt = diag(sum(Wt, 2));
dmmls = zeros(nfea, 1);

for i = 1:nfea
    mui = data(:, i)'*Dt*ones(nsmp, 1)/sum(sum(Dt));
    Lprime = (data(:, i)-mui*ones(nsmp, 1))'*L*(data(:, i)-mui*ones(nsmp, 1));
    Dprime = (data(:, i)-mui*ones(nsmp, 1))'*Dt*(data(:, i)-mui*ones(nsmp, 1));
    dmmls(i) = lamda*Lprime/Dprime;
end
[~, ranking] = sort(dmmls);
ranking = ranking';

