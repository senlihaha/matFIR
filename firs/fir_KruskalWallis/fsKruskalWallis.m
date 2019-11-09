function [ranking] = fsKruskalWallis(X, Y)

[~, n] = size(X);
out.W = zeros(n,1);

for i=1:n
    out.W(i) = -kruskalwallis(vertcat(X(:,i)', Y'),{},'off');
end

[~, ranking] = sort(out.W, 'descend');
ranking = ranking';
end