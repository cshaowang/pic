% Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In T. Dietterich,
% S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 
% (pp. 849 – 856). MIT Press.

% Asad Ali
% GIK Institute of Engineering Sciences & Technology, Pakistan
% Email: asad_82@yahoo.com

% CONCEPT: Introduced the normalization process of affinity matrix(D-1/2 A D-1/2), 
% eigenvectors orthonormal conversion and clustering by kmeans 
clear all
clc
addpath(genpath('./CalcMeasures/'));

resultdir = 'Results/';
if(~exist('Results','file'))
    mkdir('Results');
end
addpath(genpath('Results/'));
datadir = '../../Dataset/';
% add the test datasets
dataname = {'3sourceIncomplete','bbcIncomplete','bbcsportIncomplete'}; 
numdata = length(dataname);
nRepeat = 1; % runing times, set it to 20 in evaluation
MAXiter = 200; % Maximum number of iterations for KMeans 
REPlic = 10; % Number of replications for KMeans
for idata=1:numdata
    datafile = [datadir, cell2mat(dataname(idata))];
    load(datafile);
    fprintf('%s...\n',datafile);

n_views = length(data); % number of views
N = length(truelabel{1});
sigma = 1;
all_index = 1:1:N;
R = zeros(n_views, 2);
Re = [];
for v = 1:n_views
    y0 = truelabel{v};
    c = length(unique(y0));
    num = size(data{v}, 2);
    missing_index = setdiff(all_index', index{v});
    % fill missing example in both views
    fillV = mean(data{v}(:, index{v}), 2);
    for fid = 1:length(missing_index)
        data{v}(:, missing_index(fid)) = fillV;
    end
    X = data{v};
    % Normalization
    for  j = 1:num
        deno = std(X(:,j));
        if (0 == deno)
            deno = deno + eps;
        end
        X(:,j) = (X(:,j)-mean(X(:,j)))/(deno);
    end
    
    % calculate the distance
    dis = L2_distance_1(X, X);
    % calculate the affinity / similarity matrix
    A = exp(-(dis.*dis) ./ (2*sigma*sigma));
    % from affinity matrix to Laplacian matrix
    D = sum(A, 2);
    D = sqrt(1./D); % D^(-1/2)
    D = diag(D, 0);
    L = D * A * D;
    clear dis A D;
    
    % select k largest eigen vectors
    k = c;
    [eigVectors,eigValues] = eig(L);
    eigValues = diag(eigValues);
    [d1, idx] = sort(eigValues,'descend');
    nEigVec = eigVectors(:,idx(1:k));
    clear L eigVectors eigValues d1 idx;

    % construct the normalized matrix U from the obtained eigen vectors
    for i=1:size(nEigVec,1)
        normSize = sqrt(sum(nEigVec(i,:).^2));  
        if (0 == normSize)
            normSize = normSize + eps;
        end
        U(i,:) = nEigVec(i,:) ./ normSize; 
    end
    clear nEigVec normSize;
    
    % iter...
    for rtimes = 1:nRepeat
        % perform kmeans clustering on the matrix U
        [IDX, C] = kmeans(U,c,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        metric = CalcMeasures(y0, IDX);
        clear IDX;
        
        ACC(rtimes) = metric(1);
        NMI(rtimes) = metric(2);
        fprintf('=====In iteration %d=====\nACC:%.4f\tNMI:%.4f\t',rtimes,metric(1),metric(2));
    end
    R(v, 1) = mean(ACC);
    R(v, 2) = mean(NMI);
    Result(1,:) = ACC;
    Result(2,:) = NMI;
    Result(3,1) = mean(ACC);
    Result(3,2) = mean(NMI);
    Result(4,1) = std(ACC);
    Result(4,2) = std(NMI);
    save([resultdir,char(dataname(idata)),num2str(v),'_result.mat'],'Result');
    clear U ACC NMI metric Result;
end
Re = [Re; max(R(:, 1)) max(R(:, 2))];
save([resultdir,char(dataname(idata)),'_Best_result.mat'], 'Re', 'R');
clear R Re;
end
