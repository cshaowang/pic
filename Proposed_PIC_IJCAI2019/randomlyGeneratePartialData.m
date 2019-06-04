%
clc;  close all; clear all;
currentFolder = pwd;
addpath(genpath(currentFolder));
resultdir = 'GeneratedDataset/';
if(~exist('GeneratedDataset','file'))
    mkdir('GeneratedDataset');
    addpath(genpath('GeneratedDataset/'));
end
dataname = {'cornell','100leaves','NGs','HW','HW2sources','ORL','Yale','Caltech101-7'};
% dataname = {'SenITVehicle_2views_3','WikipediaArticles','WebKB_Wisconsin3_4','WebKB_Cornell3_4',...
%     'Ruters5views','Reuters','Oxford17_3view_06','Movies617','Cora4view','Cora3view',...
%     'Citeseer4view','Citeseer3view','Caltech101-7','Animals6'};
% dataname = {'Caltech101-7','Animals6','WikipediaArticles','WebKB_Cornell3_4','WebKB_Wisconsin3_4'};
n_dataset = length(dataname); % number of the datasets
for idata = 1:n_dataset
    %% read dataset
    datadir = 'Dataset/';
    dataset_file = [datadir, cell2mat(dataname(idata))];
    originalDataset = load(dataset_file);
    oriData = originalDataset.data;
    oriTruelabel = originalDataset.truelabel;
    
    N = size(oriData{1},2); % the number of instances
    n_view = length(oriData);
    perGrid = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9];
    for per_iter = 1:length(perGrid)
        per = perGrid(per_iter); % partial example ratio
        miss_n = fix(per*N); % the number of missing instances
        misingExampleVector = sort(randperm(N, miss_n));

        miss_id = 1;
        data = oriData;
        full_index = cell(1,n_view);
        for v = 1:n_view
            full_index{v} = zeros(N,1);
        end
        [~, max_id] = size(misingExampleVector);
        for id = 1:N
            if miss_id > max_id
                for v = 1:n_view
                    data{v}(:,id) = oriData{v}(:,id);
                    full_index{v}(id) = id;
                end
                continue;
            end
            if (id == misingExampleVector(miss_id))
                missingViewVector = randi([0,1],n_view,1,'int8');
                while(0 == sum(missingViewVector))
                    % in case of all views mising
                    missingViewVector = randi([0,1],n_view,1,'int8');
                end
                for j = 1:n_view
                    if 1 == missingViewVector(j)
                        data{j}(:,id) = oriData{j}(:,id);
                        full_index{j}(id) = id;
                    else
                        data{j}(:,id) = nan;
                    end
                end
                miss_id = miss_id + 1;
            else
                for k = 1:n_view
                    data{k}(:,id) = oriData{k}(:,id);
                    full_index{k}(id) = id;
                end
            end
        end
        index = cell(1, n_view);
        for v = 1:n_view
            index{v} = find(full_index{v}~=0);
        end
        truelabel = oriTruelabel;
        save([resultdir,char(dataname(idata)),'_Per',num2str(per),'.mat'],'data','truelabel','index','misingExampleVector');
        clear data truelabel index;
    end
end
