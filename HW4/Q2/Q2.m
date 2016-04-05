clc;
clear;
temp=load('ad_data.mat');
Xtrain=temp.X_train;
Xtest=temp.X_test;
ytrain=temp.y_train;
ytest=temp.y_test;
par=[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
for i=1:length(par)
[w,c]=logistic_l1_train(Xtrain, ytrain, par(i));
num_feature(i)=nnz(w);
prediction=1.0./(1.0+exp(-Xtest*w-c));
[~,~,~,AUC(i)] = perfcurve(ytest,prediction,1);
end
plot(par,AUC);
xlabel('par');
ylabel('AUC');

plot(par,num_feature);
xlabel('par');
ylabel('num_feature');