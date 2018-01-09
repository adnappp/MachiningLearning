
%ѵ��������-----
%����ÿ����ľ�ֵ�����﷢����Ȼ���������ɸ����ľ�ֵ��Э����������ɵģ�
%��������ʵ�ʵľ�ֵ��Э������󲢲�Ϊ��
function labels=bayes(testData)
%����������Ӹ�˹�ֲ������ݣ�ÿ��100����mu�������ݾ�ֵ��sigma2�������ϵ��
n=100;
mu1=[2 10];
mu2=[10 2];
sigma1=[1.5 0;0 1];
sigma2=[1 0.5;0.5 2];
r1=mvnrnd(mu1,sigma1,n);
r2=mvnrnd(mu2,sigma2,n);
%���ӻ�
subplot(1,2,1);
plot(r1(:,1),r1(:,2),'ro',r2(:,1),r2(:,2),'b*');
title('ͼ1');
fmu(:,:,1)=mean(r1);%����������ֵ��fmu1Ϊ1*2������
fmu(:,:,2)=mean(r2);
fsigma(:,:,1)=cov(r1);%��������Э����,fsigma1Ϊ2*2����
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
