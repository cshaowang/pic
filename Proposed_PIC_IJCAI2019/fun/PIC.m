function [label, weights, theta] = PIC(S, D, numC, numViews, opts)
%% input
% S: a cell where each cell (e.g., S{v}) denotes a view
% D: a cell where each cell (e.g., v-th) is denoted as D{v} = diag(sum(S{v},2))
% numC: the number of clusters
% numViews: the number of views
% opts: parameters
%% output
% label: the resulting labels
% weights: the weight of each view
% theta: canonical angles between each view and the fusion view
%%
beta = opts.beta; % beta*||w||
gamma = 0; % gamma*||w||

%% eigenvalues and the corresponding eigenvectors of each view
L = cell(1, numViews);
eigvec = cell(1, numViews);
eigval = cell(1, numViews);
U = cell(1, numViews);
for i = 1 : numViews
    Di = diag(1 ./ sqrt(diag(D{i})));
    Di((Di==inf)) = 0;
    L{i} = Di * S{i} * Di;
    [eigvec{i}, eigval{i}, U{i}] = svd(L{i});
%     [eigvec{i}, eigval{i}] = eigs(L{i}, numC, 'LM');
    eigvec{i} = eigvec{i} ./ repmat(sum(eigvec{i}, 2), 1, size(eigvec{i}, 2));
end

%% canonical angles
phi = zeros(numViews);
for i = 1 : numViews
    for j = i : numViews
        ca = svd(eigvec{i}(:, 1:numC)' * eigvec{j}(:, 1:numC));
        ca = ca/max(ca);
        phi(i ,j) = pi - max(real(acos(ca)));
        phi(j, i) = phi(i, j);
    end
end
P = diag(sum(phi));
Q = P - phi;

%% update weights (the previous method)
I = eye(numViews);
CC = zeros(numViews);
DD = zeros(numViews, 1);
for m = 1 : numViews            
    AA = eigvec{m} * eigvec{m}';
    bb = eigvec{m} * eigval{m}' * eigvec{m}';
%     bb = AA * L{m}';
    for i = 1 : numViews
        DD(i) = DD(i) + trace(L{i} * bb);
        for j = 1 : numViews
            CC(i,j) = CC(i,j) + trace(L{i} * AA * L{j}'); 
        end
    end
end
% beta = beta * norm(I + CC, 'fro') / norm( Q, 'fro');
% gamma = gamma * norm(Q + CC, 'fro') / norm(I, 'fro');
beta = beta * norm(CC, 'fro') / norm(Q, 'fro');
gamma = gamma * norm(CC, 'fro') / norm(I, 'fro');
% fprintf('...beta=%f, gamma=%f\n', beta, gamma);
H = CC + beta * Q + gamma * I ;
H = (H+H')/2;
f = -DD; 
A = -eye(numViews);
b = zeros(numViews, 1);
Aeq = ones(1, numViews);
beq = 1;
% optss = optimset('Algorithm', 'interior-point-convex');
% You may get an error on 'Active-set' due to Matlab version issue. If that, please use 'interior-point-convex'.
optss = optimset('Algorithm', 'Active-set');
weights = quadprog(H, f, A, b, Aeq, beq, [], [], [], optss);


%% the final results
Lstar = weights(1, end) * L{1};
for i = 2 : numViews
    Lstar = Lstar + weights(i, end) * L{i};
end
[V1 , ~, ~] = svd(Lstar);
V1 = V1(:, 1:numC);
VV = V1./max(repmat(sum(V1.*V1, 2).^(1/2), 1, numC), 1e-10);
[label] = kmeans(VV(:, 1:numC), numC,'Replicate', 10, 'emptyaction', 'singleton');
% [label] = litekmeans(VV(:, 2:numC), numC, 'Replicates', 50);
% [label] = k_means(V, [], numC, 100);

for i = 1 :numViews
    ca = svd(eigvec{i}(:,1:numC)' * VV);
    ca = ca/max(ca);
    theta(i) = max(real(acos(ca)));
end

end