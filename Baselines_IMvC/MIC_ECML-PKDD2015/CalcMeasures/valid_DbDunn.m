function [DB, Dunn] = valid_DbDunn(X,labels)
%==========================================================================
% FUNCTION: [DB, Dunn] = valid_DbDunn(X,labels)
% DESCRIPTION: A function for computing Davies-Bouldin index and Dunn index
%
% INPUTS:  X = a dataset, rows of X correspond to observations; columns
%              correspond to variables (exclude class labels!!)
%     labels = cluster labels from a clustering result (N-by-1 vector)
%
% OUTPUT: DB = Davies-Bouldin score
%       Dunn = Dunn score
%
% NOTE:
% Davies-Bouldin index: a low value indicates good cluster structures [Kasturi et al. 2003; Bolshakova et al. 2003].
% Dunn index: large values indicate the presence of compact and well-separated clusters [Bolshakova et al. 2003; Halkidi et al. 2001].
%==========================================================================
% Copyright (C) 2006-2007, Kaijun Wang, sunice9@yahoo.com
%==========================================================================

[nrow,nc] = size(X);
labels = (labels);
k = length(unique(labels));
[st,sw,sb,cintra,cinter] = valid_sumsqures(X,labels,k);

% +++++++ Davies-Bouldin index +++++++
R = zeros(k);
dbs=zeros(1,k);
for i = 1:k
    for j = i+1:k
        if cinter(i,j) == 0
            R(i,j) = 0;
        else
            R(i,j) = (cintra(i) + cintra(j))/cinter(i,j);
        end
    end
    dbs(i) = max(R(i,:));
end
DB = mean(dbs(1:k-1));

% ++++++++ Dunn index +++++++++
dbs = max(cintra);
R = cinter/dbs;
for i = 1:k-1
    S = R(i,i+1:k);
    dbs(i) = min(S);
end
Dunn = min(dbs);