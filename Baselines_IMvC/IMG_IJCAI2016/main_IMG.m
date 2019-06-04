clear all;
clc;
 
addpath(genpath('./misc/'));
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
    N = size(data{1},2); % the number of instances
    Re = [];
    for v1 = 1:n_view
        for v2 = v1:n_view
            if v1 == v2
                continue;
            end
            union_index = union(index{v1}, index{v2});
            joint_index = intersect(index{v1}, index{v2});
            x_index = setdiff(index{v1},joint_index);
            y_index = setdiff(index{v2},joint_index);
            all_index = 1:1:N;
            both_missing_index = setdiff(all_index', union_index);
            originalJoint = joint_index;
            joint_index = [joint_index; both_missing_index];
            % fill missing example in both views
            fillV1 = mean(data{v1}(:, index{v1}), 2);
            fillV2 = mean(data{v2}(:, index{v2}), 2);
            for fid = 1:length(both_missing_index)
                data{v1}(:, both_missing_index(fid)) = fillV1;
                data{v2}(:, both_missing_index(fid)) = fillV2;
            end
            % get the data of each view
            x_paired = data{v1}(:, joint_index);
            y_paired = data{v2}(:, joint_index);
            x_single = data{v1}(:, x_index);
            y_single = data{v2}(:, y_index);
            
            % % take truthF in case of some examples are not in both views.
            joint_n = length(joint_index);
            x_n = length(x_index);
            y_n = length(y_index);
            truth_n = joint_n + x_n + y_n;
            truthF = zeros(truth_n, 1);
            if 1 ~= size(truelabel{1}, 2)
                truelabel{1} = truelabel{1}';
            end
            truthF(1:joint_n) = truelabel{1}(joint_index, 1);
            truthF(joint_n+1:joint_n+x_n) = truelabel{1}(x_index, 1);
            truthF(joint_n+x_n+1:joint_n+x_n+y_n) = truelabel{1}(y_index, 1);
            % truthF = truelabel{1};
            numClust = length(unique(truthF));
            
            for f = 1:10 %numFold
                % Parameters for the model
                option.lamda=1e-2;
                option.beta=1;
                option.gamma=1e2;          
                option.latentdim=numClust;

                [U1 U2 P2 P1 P3 ACC(f) NMI(f) F1(f) ARI(f)] = IMGclust(x_paired',y_paired',x_single',y_single',numClust,truthF,option);
            end
            R(v1, v2, 1) = mean(ACC);
            R(v1, v2, 2) = mean(NMI);
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
            save([resultdir,char(dataname(idata)),num2str(v1),num2str(v2),'_result.mat'],'Result');
            clear ACC NMI F1 ARI metric Result;
        end
    end
    Re = [Re; max(max(R(: ,:, 1))) max(max(R(: ,:, 2)))];
    save([resultdir,char(dataname(idata)),'Best_result.mat'], 'Re', 'R');
    clear Re;
end






