function[w,b]=perceptionLearning(trainData,trainLabel,studyRate)
%trainData ѵ������
%trainLabel ѵ�����ݵı�ǩ��ȡֵΪ1��-1
%studyRate ��֪����ѧϰ��(0,1];
[nums,demens]=size(trainData);
%��ʼ��w �� b
w=zeros(1,demens);
b=0;
%ѵ������
for i=1:nums
    flag=(trainData(i,:)*w'+b)*trainLabel(i);
    if flag<=0
        w=w+studyRate*trainLabel(i)*trainData(i,:);
        b=b+studyRate*trainLabel(i);
    end
end
%���ӻ� ֻʵ����2ά�Ŀ��ӻ�
color = {'r.', 'g.', 'm.', 'b.', 'k.', 'y.'}; 
if demens==2
    subplot(1,1,1);
    plot(trainData(trainLabel==1,1),trainData(trainLabel==1,2),char(color(2)));
    hold on;
    plot(trainData(trainLabel==-1,1),trainData(trainLabel==-1,2),char(color(3)));
    hold on;
    x=linspace(0,12,5000);
    y=(-b-w(1)*x)/w(2);
    plot(x,y,'r');
    title('ѵ������');        
else
    disp('ά�Ȳ����ϻ�ͼ��׼��1ά�����û������۾Ͳ�����');
end
end