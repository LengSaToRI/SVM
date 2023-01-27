clc ;clear;close all

%%生成数据
rng('default') % 设置随机种子
grnpop = mvnrnd([1,0],0.6*eye(2),10);
redpop = mvnrnd([0,1],0.6*eye(2),10);

N=100;  %测试集个数
M=20;   %训练集个数
%可视化数据
plot(grnpop(:,1),grnpop(:,2),'go')
hold on
plot(redpop(:,1),redpop(:,2),'ro')
hold off
%由于一些红色基点接近绿色基点，因此很难仅根据位置对数据点进行分类。
%生成每个类的N个数据点。

redpts = zeros(N,2);
grnpts = redpts;
for i = 1:N
    grnpts(i,:) = mvnrnd(grnpop(randi(10),:),eye(2)*0.02);
    redpts(i,:) = mvnrnd(redpop(randi(10),:),eye(2)*0.02);
end
% 可视化数据
figure
plot(grnpts(:,1),grnpts(:,2),'go')
hold on
plot(redpts(:,1),redpts(:,2),'ro')
hold off
% 为分类准备数据
% 将数据放入一个矩阵中，并制作一个标记每个点的类别的向量grp。1表示绿色类，-1表示红色类。
cdata = [grnpts;redpts];
grp = ones(2*N,1);
grp(N+1:2*N) = -1;

%% 交叉验证
% 为交叉验证设置一个分区
c = cvpartition(2*N,'KFold',10);

% 采集函数选择EI函数，最大迭代次数设置为15
opts = struct('CVPartition',c,'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',15);
Mdl = fitcsvm(cdata,grp,'KernelFunction','rbf', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

Mdl.HyperparameterOptimizationResults.XAtMinEstimatedObjective
[x,CriterionValue,iteration] = bestPoint(Mdl.HyperparameterOptimizationResults)

%采用交叉验证分集c
L_MinEstimated = kfoldLoss(fitcsvm(cdata,grp,'CVPartition',c,'KernelFunction','rbf', ...
     'BoxConstraint',x.BoxConstraint,'KernelScale',x.KernelScale))
 
Mdl.HyperparameterOptimizationResults.XAtMinObjective
[x_observed,CriterionValue_observed,iteration_observed] = bestPoint(Mdl.HyperparameterOptimizationResults,'Criterion','min-observed')

%% 可视化优化的分类器
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(cdata(:,1)):d:max(cdata(:,1)), ...
     min(cdata(:,2)):d:max(cdata(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(Mdl,xGrid);

figure
h(1:2) = gscatter(cdata(:,1),cdata(:,2),grp,'rg','+*');
hold on
h(3) = plot(cdata(Mdl.IsSupportVector,1), ...
    cdata(Mdl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','支持向量'},'Location','Southeast');

%% 在新数据上评价准确率
% 生成新的测试点，并测试
grnobj = gmdistribution(grnpop,.2*eye(2));
redobj = gmdistribution(redpop,.2*eye(2));

newData = random(grnobj,M);
newData = [newData;random(redobj,M)];
grpData = ones(2*M,1); % green = 1
grpData(M+1:2*M) = -1; % red = -1

v = predict(Mdl,newData);
% 计算在测试数据集上的误分类率
L_Test = loss(Mdl,newData,grpData)
% 将正确分类的点标为红色方块，将错误分类的点标为黑色方块。
h(4:5) = gscatter(newData(:,1),newData(:,2),v,'mc','**');

mydiff = (v == grpData); % 标出正确分类的点
ACC=sum(mydiff)/(2*M)
for ii = mydiff
    h(6) = plot(newData(ii,1),newData(ii,2),'rs','MarkerSize',12);
end

for ii = not(mydiff)
    h(7) = plot(newData(ii,1),newData(ii,2),'ks','MarkerSize',12);
end

legend(h,{'-1 (训练集)','+1 (训练集)','支持向量', ...
    '-1 (测试集)','+1 (测试集)', ...
    '正确分类的点','错误分类的点'}, ...
    'Location','Southeast');
hold off

