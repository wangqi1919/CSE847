clc;
clear;
temp = load('USPS.mat');
A=temp.A;
[U,V]=princomp(A);
rankn=[10,50,100,200];
meanA=repmat(mean(A,1),[3000,1]);
A_nm=A-meanA;
error=[];
for i=rankn
    A1=V(:,1:i)*U(:,1:i)'+meanA;
    error=[error,norm(A-A1,'fro')];
    figure;
    A2=reshape(A1(1,:),16,16);
    imshow(A2');
    figure;
    A2=reshape(A1(2,:),16,16);
    imshow(A2');
end
