function [ w ] = leastSquares( trainData,trainLabel )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[nums,demens]=size(trainData);
trainData=[trainData,ones(nums,1)];
R=trainData'*trainData;
E=trainData'*trainLabel;
w=inv(R)*E;
%b=norm(trainLabel-trainData*w,2);

%���ӻ� ֻʵ����2ά�Ŀ��ӻ�
color = {'r.', 'g.', 'm.', 'b.', 'k.', 'y.'}; 
if demens==2
    subplot(1,1,1);
    plot(trainData(trainLabel==1,1),trainData(trainLabel==1,2),char(color(2)));
    hold on;
    plot(trainData(trainLabel==-1,1),trainData(trainLabel==-1,2),char(color(3)));
    hold on;
    x=linspace(0,12,5000);
    y=(-w(1)/w(2))*x-w(3)/w(2);
    plot(x,y,'r');
    title('ѵ������');        
else
    disp('ά�Ȳ����ϻ�ͼ��׼��1ά�����û������۾Ͳ�����');
end



end

