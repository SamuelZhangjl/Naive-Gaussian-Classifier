function [x1,x2] = twoGaussianSetGenerating(u1,u2,sigma,num1,num2)
%This function generating two gaussian set which have the same sigma

x1 = mvnrnd(u1,sigma,num1);
x2 = mvnrnd(u2,sigma,num2);


end