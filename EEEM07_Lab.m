clc;
clear all;
close all;

%% Experiment 1
%Investigate the effect of training sample size on the classifier
%performance

U1 = [3,1];
U2 = [6,4];
sigma = [1,1;1,1];

%generate design set Xd and test set Xt
[xd1,xd2] = twoGaussianSetGenerating(U1,U2,sigma,50,50);
[xt1,xt2] = twoGaussianSetGenerating(U1,U2,sigma,50,50);
xt = [xt1;xt2];

%traning samples from the design set
Nd = [5,5,10,50,100];
averageEstematorError = [0,0,0,0,0];
averageTestError = [0,0,0,0,0];
lable1_test = repmat(1,1,50);
lable2_test = repmat(2,1,50);
lable_test = [lable1_test,lable2_test];

for n = 1:5
   for v = 1:10
        [trainsamples,labels] = traningSamples(2,xd1,xd2,50,Nd(n));
        Mdl = fitcnb(trainsamples,labels,...
            'ClassNames',{'1','2'});
        estimatorError = resubLoss(Mdl,'LossFun','ClassifErr');
     	averageEstematorError(n) = averageEstematorError(n) + estimatorError;
        
        testError = loss(Mdl,xt,lable_test);
        averageTestError(n) = averageTestError(n) + testError;
   end
   averageEstematorError(n) = averageEstematorError(n)/10;
   averageTestError(n) = averageTestError(n)/10;
   disp('averageEstematorError');
   disp(averageEstematorError(n));
   disp('averageTestError');
   disp(averageTestError(n)); 
end
figure(1);
plot(Nd,averageEstematorError);
figure(2);
plot(Nd,averageTestError);

%% Experiment 2
%investigate the dependence of the Etest(Nd) curves on the dimensionality of the pattern recognition


%generate design set Xd and test set Xt (dimension = 5)
dimensions_d5 = 5;
U_d5_1 = [0,0,0,0,0];
U_d5_2 = [1,1,1,1,1];
sigma_d5 = eye(5);
[x_d5_1,x_d5_2] = twoGaussianSetGenerating(U_d5_1,U_d5_2,sigma_d5,250,250);
[x_t5_1,x_t5_2] = twoGaussianSetGenerating(U_d5_1,U_d5_2,sigma_d5,250,250);
xt_d5 = [x_t5_1;x_t5_2];

%traning samples from the design set
Nd_d5 = [10,20,50,100,200,500];
averageEstematorError_d5 = [0,0,0,0,0,0];
averageTestError_d5 = [0,0,0,0,0,0];
lable_d5_test1 = repmat(1,1,250);
lable_d5_test2 = repmat(2,1,250);
lable_test_d5 = [lable_d5_test1,lable_d5_test2];

for n = 1:6
   for v = 1:10
        [trainsamples,labels] = traningSamples(dimensions_d5,x_d5_1,x_d5_2,250,Nd_d5(n));
        Md_d5 = fitcnb(trainsamples,labels,...
            'ClassNames',{'1','2'});
        estimatorError = resubLoss(Md_d5,'LossFun','ClassifErr');
     	averageEstematorError_d5(n) = averageEstematorError_d5(n) + estimatorError;
        
        testError = loss(Md_d5,xt_d5,lable_test_d5);
        averageTestError_d5(n) = averageTestError_d5(n) + testError;
   end
   averageEstematorError_d5(n) = averageEstematorError_d5(n)/10;
   averageTestError_d5(n) = averageTestError_d5(n)/10;
   disp('averageEstematorError');
   disp(averageEstematorError_d5(n));
   disp('averageTestError');
   disp(averageTestError_d5(n)); 
end
figure(3);
plot(Nd_d5,averageEstematorError_d5);
figure(4);
plot(Nd_d5,averageTestError_d5);



%generate design set Xd and test set Xt (dimension = 10)
dimensions_d10 = 10;
U_d10_1 = [0,0,0,0,0,0,0,0,0,0];
U_d10_2 = [1,1,1,1,1,1,1,1,1,1];
sigma_d10 = eye(10);
[x_d10_1,x_d10_2] = twoGaussianSetGenerating(U_d10_1,U_d10_2,sigma_d10,250,250);
[x_t10_1,x_t10_2] = twoGaussianSetGenerating(U_d10_1,U_d10_2,sigma_d10,250,250);
xt_d10 = [x_t10_1;x_t10_2];
lable_d10_test1 = repmat(1,1,250);
lable_d10_test2 = repmat(2,1,250);
lable_test_d10 = [lable_d10_test1,lable_d10_test2];

%traning samples from the design set
Nd_d10 = [10,20,50,100,200,500];
averageEstematorError_d10 = [0,0,0,0,0,0];
averageTestError_d10 = [0,0,0,0,0,0];

for n = 1:6
   for v = 1:10
        [trainsamples,labels] = traningSamples(dimensions_d10,x_d10_1,x_d10_2,250,Nd_d10(n));
        Md_d10 = fitcnb(trainsamples,labels,...
            'ClassNames',{'1','2'});
        estimatorError = resubLoss(Md_d10,'LossFun','ClassifErr');
     	averageEstematorError_d10(n) = averageEstematorError_d10(n) + estimatorError;
        
        testError = loss(Md_d10,xt_d10,lable_test_d10);
        averageTestError_d10(n) = averageTestError_d10(n) + testError;
   end
   averageEstematorError_d10(n) = averageEstematorError_d10(n)/10;
   averageTestError_d10(n) = averageTestError_d10(n)/10;
   disp('averageEstematorError');
   disp(averageEstematorError_d10(n));
   disp('averageTestError');
   disp(averageTestError_d10(n)); 
end
figure(3);
hold on
plot(Nd_d10,averageEstematorError_d10);
hold off

figure(4);
hold on 
plot(Nd_d10,averageTestError_d10);
hold off



%generate design set Xd and test set Xt (dimension = 15)
dimensions_d15 = 15;
U_d15_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
U_d15_2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
sigma_d15 = eye(15);
[x_d15_1,x_d15_2] = twoGaussianSetGenerating(U_d15_1,U_d15_2,sigma_d15,250,250);
[x_t15_1,x_t15_2] = twoGaussianSetGenerating(U_d15_1,U_d15_2,sigma_d15,250,250);
xt_d15 = [x_t15_1;x_t15_2];

%traning samples from the design set
Nd_d15 = [10,20,50,100,200,500];
averageEstematorError_d15 = [0,0,0,0,0,0];
averageTestError_d15 = [0,0,0,0,0,0];
lable_d15_test1 = repmat(1,1,250);
lable_d15_test2 = repmat(2,1,250);
lable_test_d15 = [lable_d15_test1,lable_d15_test2];

for n = 1:6
   for v = 1:10
        [trainsamples,labels] = traningSamples(dimensions_d15,x_d15_1,x_d15_2,250,Nd_d15(n));
        Md_d15 = fitcnb(trainsamples,labels,...
            'ClassNames',{'1','2'});
        estimatorError = resubLoss(Md_d15,'LossFun','ClassifErr');
     	averageEstematorError_d15(n) = averageEstematorError_d15(n) + estimatorError;
        
        testError = loss(Md_d15,xt_d15,lable_test_d15);
        averageTestError_d15(n) = averageTestError_d15(n) + testError;
   end
   averageEstematorError_d15(n) = averageEstematorError_d15(n)/10;
   averageTestError_d15(n) = averageTestError_d15(n)/10;
   disp('averageEstematorError');
   disp(averageEstematorError_d15(n));
   disp('averageTestError');
   disp(averageTestError_d15(n)); 
end

figure(3);
hold on
plot(Nd_d15,averageEstematorError_d15);
hold off

figure(4);
hold on 
plot(Nd_d15,averageTestError_d15);
hold off


%% investigate the effect of the size of test set on the reliability of the
%%empirical error count estimator


%generate design set Xd and test set Xt (dimension = 5)
dimensions_t = 5;
U_t_1 = [0,0,0,0,0];
U_t_2 = [1,1,1,1,1];
sigma_t = eye(5);
[x_dt_1,x_dt_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,250,250);

%traning samples from the design set
Nd_dt = 200;
averageEstematorError_t = [0,0,0,0,0,0,0,0,0,0];
errorVariance_t = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t1 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t2 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t3 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t4 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t5 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t6 = [0,0,0,0,0,0,0,0,0,0];
averageTestError_t7 =  [0,0,0,0,0,0,0,0,0,0];
averageTestError_t8 =  [0,0,0,0,0,0,0,0,0,0];
averageTestError_t9 =  [0,0,0,0,0,0,0,0,0,0];
averageTestError_t10 = [0,0,0,0,0,0,0,0,0,0];

[x_tt1_1,x_tt1_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,5,5);
xt_t1 = [x_tt1_1;x_tt1_2];
lable_t1_test1 = repmat(1,1,5);
lable_t1_test2 = repmat(2,1,5);
lable_test_1 = [lable_t1_test1,lable_t1_test2];

[x_tt2_1,x_tt2_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,10,10);
xt_t2 = [x_tt2_1;x_tt2_2];
lable_t2_test1 = repmat(1,1,10);
lable_t2_test2 = repmat(2,1,10);
lable_test_2 = [lable_t2_test1,lable_t2_test2];

[x_tt3_1,x_tt3_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,20,20);
xt_t3 = [x_tt3_1;x_tt3_2];
lable_t3_test1 = repmat(1,1,20);
lable_t3_test2 = repmat(2,1,20);
lable_test_3 = [lable_t3_test1,lable_t3_test2];

[x_tt4_1,x_tt4_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,50,50);
xt_t4 = [x_tt4_1;x_tt4_2];
lable_t4_test1 = repmat(1,1,50);
lable_t4_test2 = repmat(2,1,50);
lable_test_4 = [lable_t4_test1,lable_t4_test2];

[x_tt5_1,x_tt5_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,100,100);
xt_t5 = [x_tt5_1;x_tt5_2];
lable_t5_test1 = repmat(1,1,100);
lable_t5_test2 = repmat(2,1,100);
lable_test_5 = [lable_t5_test1,lable_t5_test2];

[x_tt6_1,x_tt6_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,150,150);
xt_t6 = [x_tt6_1;x_tt6_2];
lable_t6_test1 = repmat(1,1,150);
lable_t6_test2 = repmat(2,1,150);
lable_test_6 = [lable_t6_test1,lable_t6_test2];

[x_tt7_1,x_tt7_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,200,200);
xt_t7 = [x_tt7_1;x_tt7_2];
lable_t7_test1 = repmat(1,1,200);
lable_t7_test2 = repmat(2,1,200);
lable_test_7 = [lable_t7_test1,lable_t7_test2];

[x_tt8_1,x_tt8_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,300,300);
xt_t8 = [x_tt8_1;x_tt8_2];
lable_t8_test1 = repmat(1,1,300);
lable_t8_test2 = repmat(2,1,300);
lable_test_8 = [lable_t8_test1,lable_t8_test2];

[x_tt9_1,x_tt9_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,400,400);
xt_t9 = [x_tt9_1;x_tt9_2];
lable_t9_test1 = repmat(1,1,400);
lable_t9_test2 = repmat(2,1,400);
lable_test_9 = [lable_t9_test1,lable_t9_test2];

[x_tt10_1,x_tt10_2] = twoGaussianSetGenerating(U_t_1,U_t_2,sigma_t,500,500);
xt_t10 = [x_tt10_1;x_tt10_2];
lable_t10_test1 = repmat(1,1,500);
lable_t10_test2 = repmat(2,1,500);
lable_test_10 = [lable_t10_test1,lable_t10_test2];




   for v = 1:10
        [trainsamples,labels] = traningSamples(dimensions_t,x_dt_1,x_dt_2,250,Nd_dt);
        Md_t = fitcnb(trainsamples,labels,...
            'ClassNames',{'1','2'});
        estimatorError = resubLoss(Md_t,'LossFun','ClassifErr');
     	averageEstematorError_t = averageEstematorError_t + estimatorError;
        
        testError = loss(Md_t,xt_t1,lable_test_1);
        averageTestError_t1(v) =  testError;
        
        testError = loss(Md_t,xt_t2,lable_test_2);
        averageTestError_t2(v) =  testError;
        
        testError = loss(Md_t,xt_t3,lable_test_3);
        averageTestError_t3(v) =  testError;
        
        testError = loss(Md_t,xt_t4,lable_test_4);
        averageTestError_t4(v) =  testError;
        
        testError = loss(Md_t,xt_t5,lable_test_5);
        averageTestError_t5(v) = testError;
        
        testError = loss(Md_t,xt_t6,lable_test_6);
        averageTestError_t6(v) = testError;
        
        testError = loss(Md_t,xt_t7,lable_test_7);
        averageTestError_t7(v) =  testError;
        
        testError = loss(Md_t,xt_t8,lable_test_8);
        averageTestError_t8(v) =  testError;
        
        testError = loss(Md_t,xt_t9,lable_test_9);
        averageTestError_t9(v) =  testError;
        
        testError = loss(Md_t,xt_t10,lable_test_10);
        averageTestError_t10(v) = testError;
        
   end
   averageEstematorError_t = averageEstematorError_t/10;
  
dimensionTestError = [0,0,0,0,0,0,0,0,0,0];
for t = 1:10
	dimensionTestError(1)=dimensionTestError(1)+averageTestError_t1(t);
    dimensionTestError(2)=dimensionTestError(2)+averageTestError_t2(t);
    dimensionTestError(3)=dimensionTestError(3)+averageTestError_t3(t);
    dimensionTestError(4)=dimensionTestError(4)+averageTestError_t4(t);
    dimensionTestError(5)=dimensionTestError(5)+averageTestError_t5(t);
    dimensionTestError(6)=dimensionTestError(6)+averageTestError_t6(t);
    dimensionTestError(7)=dimensionTestError(7)+averageTestError_t7(t);
    dimensionTestError(8)=dimensionTestError(8)+averageTestError_t8(t);
    dimensionTestError(9)=dimensionTestError(9)+averageTestError_t9(t);
    dimensionTestError(10)=dimensionTestError(10)+averageTestError_t10(t);
end
dimensionTestError = dimensionTestError/10;

for t = 1:10
    errorVariance_t(1)=(averageTestError_t1(t)-dimensionTestError(t))*(averageTestError_t1(t)-dimensionTestError(t));
    errorVariance_t(2)=(averageTestError_t2(t)-dimensionTestError(t))*(averageTestError_t2(t)-dimensionTestError(t));
    errorVariance_t(3)=(averageTestError_t3(t)-dimensionTestError(t))*(averageTestError_t3(t)-dimensionTestError(t));
    errorVariance_t(4)=(averageTestError_t4(t)-dimensionTestError(t))*(averageTestError_t4(t)-dimensionTestError(t));
    errorVariance_t(5)=(averageTestError_t5(t)-dimensionTestError(t))*(averageTestError_t5(t)-dimensionTestError(t));
    errorVariance_t(6)=(averageTestError_t6(t)-dimensionTestError(t))*(averageTestError_t6(t)-dimensionTestError(t));
    errorVariance_t(7)=(averageTestError_t7(t)-dimensionTestError(t))*(averageTestError_t7(t)-dimensionTestError(t));
    errorVariance_t(8)=(averageTestError_t8(t)-dimensionTestError(t))*(averageTestError_t8(t)-dimensionTestError(t));
    errorVariance_t(9)=(averageTestError_t9(t)-dimensionTestError(t))*(averageTestError_t9(t)-dimensionTestError(t));
    errorVariance_t(10)=(averageTestError_t10(t)-dimensionTestError(t))*(averageTestError_t10(t)-dimensionTestError(t));
end
errorVariance_t = errorVariance_t/9;
xindex = [5 10 20 50 100 150 200 300 400 500];
figure(5);
plot(xindex,dimensionTestError);

figure(6);
plot(xindex,errorVariance_t);














