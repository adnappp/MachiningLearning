function[w,b]=perceptionLearning(trainData,trainLabel,studyRate)
%trainData 训练数据
%trainLabel 训练数据的标签，取值为1和-1
%studyRate 感知器的学习率(0,1];
[nums,demens]=size(trainData);
%初始化w 和 b
w=zeros(1,demens);
b=0;
%训练过程
for i=1:nums
    flag=(trainData(i,:)*w'+b)*trainLabel(i);
    if flag<=0
        w=w+studyRate*trainLabel(i)*trainData(i,:);
        b=b+studyRate*trainLabel(i);
    end
end
%可视化 只实现了2维的可视化
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
    title('训练数据');        
else
    disp('维度不符合画图标准（1维的懒得画），咱就不画啦');
end
end