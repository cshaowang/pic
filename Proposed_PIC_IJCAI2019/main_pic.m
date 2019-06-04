clear all
clc
addpath(genpath(pwd))
resultdir = 'Results/';
if(~exist('Results','file'))
    mkdir('Results');
end
addpath(genpath('Results/'));
opts = [];
opts.repeat = 1; % runing times, set it to 20 in evaluation
opts.normalized = 1;
opts.beta = 0.1; % beta*w^THw;

datadir = '..\Dataset\';
% add the test datasets
dataname = {'3sourceIncomplete','bbcIncomplete','bbcsportIncomplete'}; 
numdata = length(dataname); % number of the test datasets
%% perform the proposed method on each dataset one by one...
for idata = 1:numdata
    datafile = [datadir, cell2mat(dataname(idata))];
    load(datafile);
    fprintf('%s...\n',datafile);
    view_num = length(data);
    nCluster = length(unique(truelabel{1}));

% === Normalization =======
n_num = size(data{1},2); % the number of instances
if opts.normalized
    for i = 1:view_num
        for  j = 1:n_num
            normItem = std(data{i}(:,j));
            if (0 == normItem)
                normItem = eps;
            end;
            data{i}(:,j) = (data{i}(:,j)-mean(data{i}(:,j)))/(normItem);
        end;
    end;
end
% === Similarity matrix ========
pn = 15;
W = cell(1, view_num);
distX = cell(1, view_num);
for i = 1 :view_num
    [W{i}, distX{i}] = constructS_PNG(data{i}, pn, 0);
end
% === Similarity matrix completion ===
S = SCompletion(W, index, 1);
D = cell(1, view_num);
for v = 1:view_num
    D{v} = diag(sum(S{v},2));
end
fprintf('creating similarity matrix is done\n');
% === the proposed method ===========
    ACC = zeros(opts.repeat, 1);
    NMI = zeros(opts.repeat, 1);
    for f = 1 : opts.repeat 
        [predictLabel, weight, theta] = PIC(S, D, nCluster, view_num, opts);
        metric = CalcMeasures(truelabel{1}, predictLabel);
        ACC(f) = metric(1);
        NMI(f) = metric(2);
        fprintf('\n.iteration %d, acc=%.4f, nmi=%.4f ...\n', f, ACC, NMI);
    end
    Result(1,:) = ACC;
    Result(2,:) = NMI;
    Result(3,1) = mean(ACC);
    Result(3,2) = mean(NMI);
    Result(4,1) = std(ACC);
    Result(4,2) = std(NMI);
    save([resultdir,char(dataname(idata)),'_result.mat'],'Result');
    clear ACC NMI metric Result;
end