function [ svm ] = svmTrain( trainData,trainLabel,kertype,C )
options=optimset;
options.LargerScale='off';
options.Display='off';

n=length(trainLabel);
H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
f=-ones(n,1);
A=[];
b=[];
Aeq=trainLabel;
beq=0;
lb=zeros(n,1);
ub=C*ones(n,1);
a0=zeros(n,1);
[a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
epsilon=1e-8;
sv_label=find(abs(a)>epsilon);
svm.a=a(sv_label);
svm.Xsv=trainData(:,sv_label);
svm.Ysv=trainLabel(sv_label);
svm.svnum=length(sv_label);
end

