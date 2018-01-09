clear;
clc;
n=100;
%生成训练数据
mu1=[2 10];
mu2=[10 2];
sigma1=[1.5 0;0 1];
sigma2=[1 0.5;0.5 2];
r1=mvnrnd(mu1,sigma1,n);
r2=mvnrnd(mu2,sigma2,n);
trainData=[r1;r2];
trainLabel=[ones(100,1);2*ones(100,1)];
%生成测试数据
mu1=[5 5];
n=200;
sigma1=[0.5 0;0 0.5];
testData=mvnrnd(mu1,sigma1,n);
%测试
labels=bayesClassifer(trainData,trainLabel,2,testData);