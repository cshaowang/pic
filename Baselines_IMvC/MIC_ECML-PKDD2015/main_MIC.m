clear all;
clc;

addpath(genpath('./CalcMeasures/'));

resdir='data/result/';
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
    n_view = length(data);
    N = size(data{1}, 2); % the number of instances

    label = truelabel{1}';
    K = length(unique(label));
    for v = 1:n_view;
       options.alpha(v) = 0.001;
       options.beta(v) = 0.001;
    end

    options.rounds = 50;
    options.error = 1e-4;
    options.maxIter = 10;
    options.nRepeat = 1;
    options.minIter = 50;
    options.meanFitRatio = 0.1;
    options.kmeans = 1;

    [XX, W] = Fill_missing_data(data, index);
    for i = 1:size(XX,2)
        XX{1, i} = XX{1, i}';    
    end
    for f = 1:10
        [U, V, centroidU, log, ACC(f), NMI(f), F1(f), ARI(f), time] = MultiNMF_incomplete_original_l21(XX, W, K, label, options);
    end
    Result(1,:) = ACC;
    Result(2,:) = NMI;
    Result(3,:) = F1;
    Result(4,:) = ARI;
    Result(5,1) = mean(ACC);
    Result(5,2) = mean(NMI);
    Result(5,3) = mean(F1);
    Result(5,4) = mean(ARI);
    Result(6,1) = std(ACC);
    Result(6,2) = std(NMI);
    Result(6,3) = std(F1);
    Result(6,4) = std(ARI);
    save([resultdir,char(dataname(idata)),'_result.mat'],'Result');
    clear ACC NMI F1 ARI metric Result;
end