function [w]=Ridge(x,y,lambda)
w=(x'*x+lambda*eye(size(x'*x)))^-1*x'*y;
end