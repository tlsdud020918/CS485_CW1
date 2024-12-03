clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

% EXPERIMENT VARIABLES
splitnum_list = [1,2,3,4,5,10,20,30,40,50,100,250,500];
%splitnum_list = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,200];
num_repeats = 5;
% EXPERIMENT VARIABLES END

% DEFAULT PARAMETERS
param.num = 250;
param.depth = 8;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = 'axis-aligned';
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

results.splitNum=splitnum_list;

% -------------- EXPERIMENT FOR LOOP START ------------------- %
%% 2. Experiment with splitnum using best num_trees and depth_trees from previous step
trainacc_matrix = zeros(1, length(splitnum_list));
testacc_matrix = zeros(1, length(splitnum_list));
train_time_matrix = zeros(1, length(splitnum_list));
test_time_matrix = zeros(1, length(splitnum_list));
best_test_result = struct();
worst_test_result = struct();
best_train_result = struct();
worst_train_result = struct();

for i = 1:length(splitnum_list)
    fprintf("# splits: %i\n", i);

    train_acc_list = [];
    test_acc_list = [];
    train_time_list = [];
    test_time_list = [];
    train_results = {};
    test_results = {};

    % Run experiment num_repeats times for each configuration
    for k = 1:num_repeats
        [train_acc, test_acc, train_time, test_time, train_result, test_result] = RF_function(param.num, param.depth, splitnum_list(i), 'axis-aligned', false);
        train_acc_list(end+1) = train_acc;
        test_acc_list(end+1) = test_acc;
        train_time_list(end+1) = train_time;
        test_time_list(end+1) = test_time;
        train_results{k} = train_result;
        test_results{k} = test_result;

        % Calculate and store averages
        trainacc_matrix(i) = mean(train_acc_list);
        testacc_matrix(i) = mean(test_acc_list);
        train_time_matrix(i) = mean(train_time_list);
        test_time_matrix(i) = mean(test_time_list);
    
        % Save best and worst case test results
        [~, best_idx] = max(test_acc_list);
        [~, worst_idx] = min(test_acc_list);
        best_test_result.(['split_', num2str(splitnum_list(i))]) = test_results{best_idx};
        worst_test_result.(['split_', num2str(splitnum_list(i))]) = test_results{worst_idx};
    
        % Save best and worst case train results
        [~, best_idx] = max(train_acc_list);
        [~, worst_idx] = min(train_acc_list);
        best_train_result.(['split_', num2str(splitnum_list(i))]) = train_results{best_idx};
        worst_train_result.(['split_', num2str(splitnum_list(i))]) = train_results{worst_idx};
    
        % Save intermediate results after processing each depth
        results.num_depth_trainaccuracy = trainacc_matrix;
        results.num_depth_testaccuracy = testacc_matrix;
        results.num_depth_train_time = train_time_matrix;
        results.num_depth_test_time = test_time_matrix;
        results.best_test_result = best_test_result;
        results.worst_test_result = worst_test_result;
        results.best_train_result = best_train_result;
        results.worst_train_result = worst_train_result;
        save(['./result/q2-2 (num=250,depth=8)/result_',num2str(splitnum_list(i)),'.mat'], 'results');
    end
end

