function [weight]= logistic_train(weight,data,labels,e,maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
y_new=caly(data,weight);
[~,d]=size(data);
for iter=1:maxiter
  y_old=y_new;
  Rz=calRnn(y_old)*data*weight-(y_old-labels);
  weight=(data'*calRnn(y_old)*data)\data'*Rz;
  y_new=caly(data,weight);
  if sum(abs(y_new-y_old))/d<e
      break;
  end
end
end
function value=caly(X,w)
value=1.0./(1.0+exp(-X*w));
end
function Rnn=calRnn(y)
Rnn=diag(y.*(1.0-y));
end