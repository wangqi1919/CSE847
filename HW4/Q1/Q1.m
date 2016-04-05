clc;
clear;
data=xlsread('data.xlsx');
labels=xlsread('labels.xlsx');
data=[data,ones(size(data,1),1)];
Xtrain=data(1:2000,:);
Xtest=data(2001:end,:);
Ytrain=labels(1:2000,:);
Ytest=labels(2001:end,:);
tol=1e-5;
maxiter=1000;
n=[200,500,800,1000,1500,2000];

for i=1:length(n)
    data=Xtrain(1:n(i),:);
    labels=Ytrain(1:n(i),:);
    w0=zeros(size(data,2),1);
   w=logistic_train(w0,data,labels,tol,maxiter);
   prediction=1.0./(1.0+exp(-Xtest*w));
   prediction(prediction>=0.5)=1;
   prediction(prediction<0.5)=0;
   accuracy(i)=sum(prediction==Ytest)/length(Ytest);
   
end
plot(n,accuracy);
xlabel('n');
ylabel('accuracy');

