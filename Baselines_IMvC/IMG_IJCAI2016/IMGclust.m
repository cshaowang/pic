function [Ux Uy P2 P1 P3 Acc nmi F1 ARI] = IMGclust(X2,Y2,X1,Y3,numClust,truth,option)

% Partial codes are from 
% S.-Y. Li, Y. Jiang nd Z.-H. Zhou. Partial Multi-View Clustering. In: Proceedings of the 28th AAAI Conference on
% Artificial Intelligence (AAAI'14),Qu��bec, Canada ,2014.

% Contact handong.zhao@gmail.com if you have any questions
% Zhao, et al., Incomplete Multi-modal Visual Data Grouping, IJCAI'16

%%
if (min(truth)==0)
    truth = truth + 1;
end

option.option = numClust;
option.truth = truth;
[Ux,Uy,P2,P1,P3,W] = IMG(X2,Y2,X1,Y3,option);

% fprintf('running spectral clustering...\n');
kmeans_avg_iter = 10;
for i=1: kmeans_avg_iter

    C = clu_ncut(W,numClust);
    C = C';
    
    %%
    metric = CalcMeasures(truth, C);
    Acci(i) = metric(1);
    nmii(i) = metric(2);
    Fi(i) = metric(3);
    ARii(i) = metric(5);
end
Acc = mean(Acci);
nmi = mean(nmii);
F1 = mean(Fi);
ARI = mean(ARii);

fprintf('acc: %f(%f)\tnmi: %f(%f)\n', Acc, std(Acci), nmi, std(nmii));

