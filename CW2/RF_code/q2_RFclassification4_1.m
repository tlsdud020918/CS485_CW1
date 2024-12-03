clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

% EXPERIMENT VARIABLES
num_trees_list = [1,5,10,25,50,100,250,500];
depth_trees_list = [2, 4, 6, 8, 10, 12];
num_repeats = 5;
% EXPERIMENT VARIABLES END

% DEFAULT PARAMETERS
param.num = 100;
param.depth = 6;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = 'two-pixel';
% DEFAULT PARAMETERS END

% DATA PARAMETERS (FIXED)
global vocab_size is_kmeans showImg feature_save vocab_save;
vocab_size = 256;
is_kmeans = 1;
showImg = 0;
feature_save = 0;
vocab_save = 0;
% DATA PARAMETERS END (FIXED)

[data_train, data_test] = getData('Caltech'); 
disp(size(data_train))
disp(size(data_test))

results.tree_nums=num_trees_list;
results.tree_depths=depth_trees_list;

% -------------- EXPERIMENT FOR LOOP START ------------------- %
%% 1. Experiment with num_trees and depth_trees
trainacc_matrix = zeros(length(num_trees_list), length(depth_trees_list));
testacc_matrix = zeros(length(num_trees_list), length(depth_trees_list));
train_time_matrix = zeros(length(num_trees_list), length(depth_trees_list));
test_time_matrix = zeros(length(num_trees_list), length(depth_trees_list));
best_test_result = struct();
worst_test_result = struct();
best_train_result = struct();
worst_train_result = struct();

for i = 1:length(num_trees_list)
    fprintf("# of trees: %i\n", num_trees_list(i));
    for j = 1:length(depth_trees_list)
        train_acc_list = [];
        test_acc_list = [];
        train_time_list = [];
        test_time_list = [];
        train_results = {};
        test_results = {};

        % Run experiment num_repeats times for each configuration
        for k = 1:num_repeats
            [train_acc, test_acc, train_time, test_time, train_result, test_result] = RF_function(num_trees_list(i), depth_trees_list(j), param.splitNum, param.weaklearner, false);
            train_acc_list(end+1) = train_acc;
            test_acc_list(end+1) = test_acc;
            train_time_list(end+1) = train_time;
            test_time_list(end+1) = test_time;
            train_results{k} = train_result;
            test_results{k} = test_result;
        end

        % Calculate and store averages
        trainacc_matrix(i, j) = mean(train_acc_list);
        testacc_matrix(i, j) = mean(test_acc_list);
        train_time_matrix(i, j) = mean(train_time_list);
        test_time_matrix(i, j) = mean(test_time_list);

        % Save best and worst case test results
        [~, best_idx] = max(test_acc_list);
        [~, worst_idx] = min(test_acc_list);
        best_test_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = test_results{best_idx};
        worst_test_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = test_results{worst_idx};
        
        % Save best and worst case train results
        [~, best_idx] = max(train_acc_list);
        [~, worst_idx] = min(train_acc_list);
        best_train_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = train_results{best_idx};
        worst_train_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = train_results{worst_idx};

        % Save intermediate results after processing each depth
        results.num_depth_trainaccuracy = trainacc_matrix;
        results.num_depth_testaccuracy = testacc_matrix;
        results.num_depth_train_time = train_time_matrix;
        results.num_depth_test_time = test_time_matrix;
        results.best_test_result = best_test_result;
        results.worst_test_result = worst_test_result;
        results.best_train_result = best_train_result;
        results.worst_train_result = worst_train_result;
        save(['./result/q2-1 (split=10,two-pixel)/result_', num2str(i), '_', num2str(j), '.mat'], 'results');
    end
end
