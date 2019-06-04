function [acc, nmi, f1, ari] = printResult(X, label, K, kmeansFlag)

if kmeansFlag == 1
    indic = litekmeans(X, K, 'Replicates',20);
else
    [~, indic] = max(X, [] ,2);
end
metric = CalcMeasures(label, indic);
acc = metric(1);
nmi = metric(2);
f1 = metric(3);
ari = metric(5);
disp(sprintf('acc: %0.4f\tnmi:%.4f', acc, nmi));
end