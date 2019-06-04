function [XX, W] = Fill_missing_data(X, index)

n_views = size(X, 2); % the number of views
N = size(X{1, 1}, 2); % the number of instance

%%========================= missing information ====================
all_index = 1:1:N;
missing_index = cell(1, n_views);
for v = 1:n_views
    missing_index{v} = setdiff(all_index', index{v});
end

%%========================= Index matrix ===========================
M = zeros(n_views, N);
for v = 1:n_views
    M(v, index{v}) = 1;
end

%======================== diagonal weight matrix ===================
MM = sum(M, 2)/N;
W = cell(1, n_views);

for v = 1:n_views
    W{1, v} = ones(N, 1);
    W{1, v}(index{v}, 1) = MM(v);
    W{1, v} = diag(W{1, v});
end
           
%%================= fill the missing example with average feature ========
for v = 1:n_views
    aver = mean(X{1, v}(:, index{v}), 2);
    for ii = 1:length(missing_index{v})
        fillid = missing_index{v}(ii);
        X{1, v}(:, fillid) = aver;
    end
end

%%------------------------------------------------------------
% XX=cell(1,num_views);
% for v=1:num_views
%     X{1,v}=X{1,v}';
%     XX{1,v}=W{1,v}*X{1,v};
%     XX{1,v}=XX{1,v}';
% end
XX = X;

end

