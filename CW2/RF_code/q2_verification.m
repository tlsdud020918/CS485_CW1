clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

param.num = 100;
param.depth = 5;    % trees depth
param.splitNum = 3; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = 'axis-aligned';
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

% data load
[data_train, data_test] = getData('Caltech'); 
disp(size(data_train))
disp(size(data_test))

% Train Random Forest
tic;
trees = growTrees(data_train, param);
train_time = toc;

% Test Random Forest
tic;
testTrees_script;
test_time = toc;

%test_result = c;

% Result
% 1. accuracy
accuracy = accuracy_rf;

% 2. confusioin matrix
%C = confusionmat(categorical(test_L), categorical(test_result));
%confusionchart(C);

% 3. example success/failures ??? -> 구상 필요
