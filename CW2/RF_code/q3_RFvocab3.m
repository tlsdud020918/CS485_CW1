clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

% EXPERIMENT VARIABLES
splitnum_list = [1,5,10,30,50,100,250,500];
num_repeats = [5,5,5,5,5,1,1,1];
% EXPERIMENT VARIABLES END

% DEFAULT PARAMETERS
param.num = 250;
param.depth = 8;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = 'axis-aligned';
% DEFAULT PARAMETERS END

% DATA PARAMETERS (FIXED)
global vocab_size is_kmeans showImg feature_save vocab_save vocab_param;
global vocab_time train_hist_time test_hist_time;
vocab_size = 256;
is_kmeans = 0;
showImg = 0;
feature_save = 0;
vocab_save = 1;

vocab_param.num = 8; % prior experiment: 전체 256에 이거 32
vocab_param.depth = 6;    % trees depth
vocab_param.splitNum = 10; % Number of trials in split function
vocab_param.split = 'IG'; % Currently support 'information gain' only
vocab_param.weaklearner = 'axis-aligned';
% DATA PARAMETERS END (FIXED)

results.splitNum = splitnum_list;

% -------------- EXPERIMENT FOR LOOP START ------------------- %
%% 2. Experiment with splitnum using best num_trees and depth_trees from previous step
trainacc_matrix = zeros(1, length(splitnum_list));
testacc_matrix = zeros(1, length(splitnum_list));
vocab_time_matrix = zeros(1, length(splitnum_list));
trquan_time_matrix = zeros(1, length(splitnum_list));
tequan_time_matrix = zeros(1, length(splitnum_list));
best_test_result = struct();
worst_test_result = struct();
best_train_result = struct();
worst_train_result = struct();
vocab_size_matrix = zeros(1, length(splitnum_list));

for i = 1:length(splitnum_list)
    fprintf("# splits: %i\n", i);

    train_acc_list = [];
    test_acc_list = [];
    vocab_time_list = [];
    trquan_time_list = [];
    tequan_time_list = [];
    train_results = {};
    test_results = {};
    vocab_size_list = [];
        
    % VARIABLE SETTING %
    vocab_param.splitNum = splitnum_list(i);
    % VARIABLE SETTING END %

    % Run experiment num_repeats times for each configuration
    for k = 1:num_repeats(i)
        [train_acc, test_acc, ~, ~, train_result, test_result] = RF_function(param.num, param.depth, param.splitNum, param.weaklearner, false);
        train_acc_list(end+1) = train_acc;
        test_acc_list(end+1) = test_acc;
        train_results{k} = train_result;
        test_results{k} = test_result;
            
        trquan_time = train_hist_time;
        tequan_time = test_hist_time;
            
        vocab_time_list(end+1) = vocab_time;
        trquan_time_list(end+1) = trquan_time;
        tequan_time_list(end+1) = tequan_time;
        vocab_size_list(end+1) = vocab_size;

        % Calculate and store averages
        trainacc_matrix(i) = mean(train_acc_list);
        testacc_matrix(i) = mean(test_acc_list);
    
        vocab_time_matrix(i) = mean(vocab_time_list);
        trquan_time_matrix(i) = mean(trquan_time_list);
        tequan_time_matrix(i) = mean(tequan_time_list);
        vocab_size_matrix(i) = mean(vocab_size_list);
    
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
        results.vocab_time = vocab_time_matrix;
        results.train_quantization_time = trquan_time_matrix;
        results.test_quantization_time = tequan_time_matrix;
        results.best_test_result = best_test_result;
        results.worst_test_result = worst_test_result;
        results.best_train_result = best_train_result;
        results.worst_train_result = worst_train_result;
        results.vocab_size = vocab_size_matrix;
        save(['./result/q3-2(1)/result_', num2str(splitnum_list(i)),'.mat'], 'results');
    end
end
