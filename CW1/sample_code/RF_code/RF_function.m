function [accuracy, test_result, time] = RF_function(num_trees, depth_trees, splitnum, weaklearner, conf_mtx)

param.num = num_trees;
param.depth = depth_trees;    % trees depth
param.splitNum = splitnum; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = weaklearner;
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

% data load
file = load("../../dataset/face_split.mat");
train_X = file.train_X;
train_L = file.train_L;
test_X = file.test_X;
test_L = file.test_L;

data_train = cat(2, transpose(train_X), transpose(train_L));
data_test = cat(2, transpose(test_X), transpose(test_L));

% Train Random Forest
tic;
trees = growTrees(data_train, param);
train_time = toc;

% Test Random Forest
tic;
testTrees_script;
test_time = toc;

test_result = c;
time = [train_time test_time];

% Result
% 1. accuracy
accuracy = accuracy_rf;

% 2. confusioin matrix
if conf_mtx
    C = confusionmat(categorical(test_L), categorical(test_result));
    confusionchart(C);
end

% 3. example success/failures ??? -> 구상 필요
