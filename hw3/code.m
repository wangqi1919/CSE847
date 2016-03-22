clc;
clear;
temp=load('diabetes.mat');
xtrain=temp.x_train;
ytrain=temp.y_train;
xtest=temp.x_test;
ytest=temp.y_test;
%%
lambda=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
for i=1:length(lambda)
    w=Ridge(xtrain,ytrain,lambda(i));
    trainerror(i)=calMSE(xtrain*w,ytrain);
    testerror(i)=calMSE(xtest*w,ytest);
end

%%
sizev=length(ytrain)-round(length(ytrain)/5);
for i=1:5
    rng(i)
    id=randperm(length(ytrain));
    xtraintrain=xtrain(1:sizev,:);
    xtraintest=xtrain(sizev+1:end,:);
    ytraintrain=ytrain(1:sizev,:);
    ytraintest=ytrain(sizev+1:end,:);
    for j=1:length(lambda)
        w=Ridge(xtraintrain,ytraintrain,lambda(j));
        traintesterror(i,j)=calMSE(xtraintest*w,ytraintest);   
    end
    
end
crosserror=mean(traintesterror,1);
%%
x=[-5,-4,-3,-2,-1,0,1];
y1=trainerror;
y2=testerror;
y3=crosserror;
plot(x,y1,'+b-',x,y2,'*k-',x,y3,'or-');
legend('train error','test error','cross validation error');



