clear all;
clc;

addpath(genpath('./CalcMeasures/'));

resultdir = 'Results/';
if(~exist('Results','file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end
datasetdir = '../../Dataset/';
dataname = {'3sourceIncomplete','bbcIncomplete','bbcsportIncomplete'};
numdata = length(dataname); % data number
for idata = 1:numdata
    datafile = [datasetdir, cell2mat(dataname(idata))];
    load(datafile);
    viewNum = length(data);
    N = size(data{1}, 2); % the number of instances
    
    % data preprocessing
    all_index = 1:1:N;
    missing_index = cell(1, viewNum);
    W = cell(viewNum, 1);
    X = cell(viewNum, 1);
    for v = 1:viewNum
        ww = ones(N, 1);
        missing_index{v} = setdiff(all_index', index{v});
        data{v}(:, missing_index{v}) = 0;
        X{v} = NormalizeFea(data{v}, 0);
        ww(missing_index{v}, 1) = 0;
        W{v} = diag(ww);
    end

    label = truelabel{1};
    latentDim = length(unique(label));
    nRepeat = 10;
    ACC = zeros(nRepeat, 1);
    NMI = zeros(nRepeat, 1);
    for f = 1:nRepeat
        r = latentDim;
        options.afa = 1e1;
        options.beta = 1e0;
        disp([options.afa, options.beta]);
        [U, V, B, F, P, N] = DAIMC(X, W, label, r, viewNum, options);
        metric = printResult(V, label, r, 1);
        ACC(f) = metric(1);
        NMI(f) = metric(2);
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