function [ w ] = leastSquares( trainData,trainLabel )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[nums,demens]=size(trainData);
trainData=[trainData,ones(nums,1)];
R=trainData'*trainData;
E=trainData'*trainLabel;
w=inv(R)*E;
%b=norm(trainLabel-trainData*w,2);

%可视化 只实现了2维的可视化
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
    title('训练数据');        
else
    disp('维度不符合画图标准（1维的懒得画），咱就不画啦');
end



end

