clc;
clear;
C=10;
kertype='linear';
%���ɲ�������
n=30;
x1=randn(2,n);
x2=4+randn(2,n);

x3=randn(2,n);
x3=[x3(1,:)+4;x3(2,:)-4];

x4=randn(2,n);
x4=[x4(1,:)+8;x4(2,:)];
%���ӻ���������
plot(x1(1,:),x1(2,:),'bs',x2(1,:),x2(2,:),'k+');
hold on;
plot(x3(1,:),x3(2,:),'r*',x4(1,:),x4(2,:),'y.');
axis([-3 11 -7 7]);
hold on;
%�����ϳ�һ��ѵ����ѵ��ģ��
trainData12=[x1,x2];
trainData13=[x1,x3];
trainData14=[x1,x4];
trainData23=[x2,x3];
trainData24=[x2,x4];
trainData34=[x3,x4];
trainLabel=[ones(1,n),-ones(1,n)];

svm_12=svmTrain(trainData12,trainLabel,kertype,C);
svm_13=svmTrain(trainData13,trainLabel,kertype,C);
svm_14=svmTrain(trainData14,trainLabel,kertype,C);
svm_23=svmTrain(trainData23,trainLabel,kertype,C);
svm_24=svmTrain(trainData24,trainLabel,kertype,C);
svm_34=svmTrain(trainData34,trainLabel,kertype,C);

%���ɲ�������
[x1,x2] = meshgrid(-2:0.05:10,-6:0.05:6);  %x1��x2����181*181�ľ���  
[rows,cols] = size(x1);    
nt = rows*cols;                    
Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];  
%ǰ���reshape(x1,1,nt)�ǽ�x1ת��1*��181*181���ľ�������xt��2*��181*181���ľ���  
%reshape�������µ���������С��С�ά��  
Yt = ones(1,nt);  
result12=svmTest(svm_12,Xt,Yt,kertype);
Yd = reshape(result12.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���  
hold on;
result13=svmTest(svm_13,Xt,Yt,kertype);
Yd = reshape(result13.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���  
hold on;
result14=svmTest(svm_14,Xt,Yt,kertype);
Yd = reshape(result14.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���  
hold on;
result23=svmTest(svm_23,Xt,Yt,kertype);
Yd = reshape(result23.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���  
hold on;
result24=svmTest(svm_24,Xt,Yt,kertype);
Yd = reshape(result24.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���  
hold on;
result34=svmTest(svm_34,Xt,Yt,kertype);
Yd = reshape(result34.Y,rows,cols); 
contour(x1,x2,Yd,[0,0],'ShowText','on');%���ȸ���

%����һ��������������һ��
Xt=[10;2];
Yt=1;
result12=svmTest(svm_12,Xt,Yt,kertype);
result13=svmTest(svm_13,Xt,Yt,kertype);
result14=svmTest(svm_14,Xt,Yt,kertype);
result23=svmTest(svm_23,Xt,Yt,kertype);
result24=svmTest(svm_24,Xt,Yt,kertype);
result34=svmTest(svm_34,Xt,Yt,kertype);
if result12.Y==1&&result13.Y==1&&result14.Y==1
    testLabel=1;
elseif result12.Y==-1&&result23.Y==1&&result24.Y==1
    testLabel=2;
elseif result13.Y==-1&&result23.Y==-1&&result34.Y==1
    testLabel=3;
elseif result14.Y==-1&&result24.Y==-1&&result34.Y==-1
    testLabel=4;
else
    testLabel=-1;
    disp('���Ե㲻������4����');
end