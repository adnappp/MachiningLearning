
%训练分类器-----
%计算每个类的均值，这里发现虽然样本点是由给出的均值和协方差矩阵生成的，
%但是样本实际的均值和协方差矩阵并不为它
function labels=bayes(testData)
%生成两类服从高斯分布的数据，每类100个，mu代表数据均值，sigma2代表相关系数
n=100;
mu1=[2 10];
mu2=[10 2];
sigma1=[1.5 0;0 1];
sigma2=[1 0.5;0.5 2];
r1=mvnrnd(mu1,sigma1,n);
r2=mvnrnd(mu2,sigma2,n);
%可视化
subplot(1,2,1);
plot(r1(:,1),r1(:,2),'ro',r2(:,1),r2(:,2),'b*');
title('图1');
fmu(:,:,1)=mean(r1);%计算样本均值，fmu1为1*2行向量
fmu(:,:,2)=mean(r2);
fsigma(:,:,1)=cov(r1);%计算样本协方差,fsigma1为2*2矩阵
fsigma(:,:,2)=cov(r2);
[m,~]=size(testData);
labels=zeros(1,m);
for i=1:m
    P=zeros(1,2);
    for j=1:2
        P(j)=mvnpdf(testData(i,:),fmu(:,:,j),fsigma(:,:,j));
    end
    if P(1)>P(2)
        labels(i)=1;
    else
        labels(i)=2;
    end
end
end
