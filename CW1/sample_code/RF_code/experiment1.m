init ;

% Parameters for experiments
num_trees_list = [1,5,10,25,50,100,250,500];
depth_trees_list = [2, 4, 6, 8, 10, 12]; %, 14, 16
splitnum_list = [1,25,50,100,150,300,500];
weaklearner_list = {'axis-aligned', 'two-pixel', 'linear', 'non-linear'};

% Store results
results = struct();

% Number of repetitions for each parameter configuration
num_repeats = 10;

pause;

%% 1. Experiment with num_trees and depth_trees
accuracy_matrix = zeros(length(num_trees_list), length(depth_trees_list));
train_time_matrix = zeros(length(num_trees_list), length(depth_trees_list));
test_time_matrix = zeros(length(num_trees_list), length(depth_trees_list));
best_test_result = struct();
worst_test_result = struct();

for i = 1:length(num_trees_list)
    fprintf("# of trees: %i\n", i);
    for j = 1:length(depth_trees_list)
        acc_list = [];
        train_time_list = [];
        test_time_list = [];
        test_results = {};
        
        % Run experiment 10 times for each configuration
        for k = 1:num_repeats
            [accuracy, train_time, test_time, test_result] = RF_function(num_trees_list(i), depth_trees_list(j), 10, 'axis-aligned', false);
            acc_list(end+1) = accuracy;
            train_time_list(end+1) = train_time;
            test_time_list(end+1) = test_time;
            test_results{k} = test_result;
        end
        
        % Calculate and store averages
        accuracy_matrix(i, j) = mean(acc_list);
        train_time_matrix(i, j) = mean(train_time_list);
        test_time_matrix(i, j) = mean(test_time_list);
        
        % Save best and worst case test results
        [~, best_idx] = max(acc_list);
        [~, worst_idx] = min(acc_list);
        best_test_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = test_results{best_idx};
        worst_test_result.(['num_', num2str(num_trees_list(i)), '_depth_', num2str(depth_trees_list(j))]) = test_results{worst_idx};
    end
end

% Save intermediate results
results.num_depth_accuracy = accuracy_matrix;
results.num_depth_train_time = train_time_matrix;
results.num_depth_test_time = test_time_matrix;
results.best_test_result = best_test_result;
results.worst_test_result = worst_test_result;
save('result.mat', 'results');  % Save after each experiment

% Plot for num_trees and depth_trees
figure;
subplot(1,3,1);
surf(num_trees_list, depth_trees_list, accuracy_matrix');
title('Accuracy for num\_trees vs depth\_trees');
xlabel('Number of Trees');
ylabel('Tree Depth');
zlabel('Accuracy');

subplot(1,3,2);
surf(num_trees_list, depth_trees_list, train_time_matrix');
title('Training Time for num\_trees vs depth\_trees');
xlabel('Number of Trees');
ylabel('Tree Depth');
zlabel('Training Time (s)');

subplot(1,3,3);
surf(num_trees_list, depth_trees_list, test_time_matrix');
title('Testing Time for num\_trees vs depth\_trees');
xlabel('Number of Trees');
ylabel('Tree Depth');
zlabel('Testing Time (s)');
saveas(gcf, 'Graph1.png');

pause;

%% 2. Experiment with splitnum using best num_trees and depth_trees from previous step
%[~, max_idx] = max(accuracy_matrix(:));
%[best_num_idx, best_depth_idx] = ind2sub(size(accuracy_matrix), max_idx);
best_num_trees = 250;
best_depth_trees = 8;

splitnum_list = [1,10,25,50,100,250,500];
num_repeats = 1;

split_accuracy = zeros(1, length(splitnum_list));
split_train_time = zeros(1, length(splitnum_list));
split_test_time = zeros(1, length(splitnum_list));
split_best_test_result = struct();
split_worst_test_result = struct();

for i = 1:length(splitnum_list)
    fprintf("# splits: %i\n", i);
    acc_list = [];
    train_time_list = [];
    test_time_list = [];
    test_results = {};
    
    % Run experiment 10 times for each splitnum
    for k = 1:num_repeats
        [accuracy, train_time, test_time, test_result] = RF_function(best_num_trees, best_depth_trees, splitnum_list(i), 'axis-aligned', false);
        acc_list(end+1) = accuracy;
        train_time_list(end+1) = train_time;
        test_time_list(end+1) = test_time;
        test_results{k} = test_result;
    end
    
    % Calculate and store averages
    split_accuracy(i) = mean(acc_list);
    split_train_time(i) = mean(train_time_list);
    split_test_time(i) = mean(test_time_list);
    
    % Save best and worst case test results
    [~, best_idx] = max(acc_list);
    [~, worst_idx] = min(acc_list);
    split_best_test_result.(['split_', num2str(splitnum_list(i))]) = test_results{best_idx};
    split_worst_test_result.(['split_', num2str(splitnum_list(i))]) = test_results{worst_idx};
end

results.split_accuracy = split_accuracy;
results.split_train_time = split_train_time;
results.split_test_time = split_test_time;
results.split_best_test_result = split_best_test_result;
results.split_worst_test_result = split_worst_test_result;
save('result.mat', 'results', '-append');  % Save after each experiment

% Plot for splitnum
figure;
subplot(1,3,1);
plot(splitnum_list, split_accuracy, '-o');
title('Accuracy for splitnum');
xlabel('Splitnum');
ylabel('Accuracy');

subplot(1,3,2);
plot(splitnum_list, split_train_time, '-o');
title('Training Time for splitnum');
xlabel('Splitnum');
ylabel('Training Time (s)');

subplot(1,3,3);
plot(splitnum_list, split_test_time, '-o');
title('Testing Time for splitnum');
xlabel('Splitnum');
ylabel('Testing Time (s)');
saveas(gcf, 'Graph2.png');

%% 3. Experiment with weaklearner using best cases from previous steps
num_repeats=5;
best_num_trees = 250;
best_depth_trees = 8;
weaklearner_list = {'two-pixel'};
weaklearner_accuracy = zeros(1, length(weaklearner_list));
weaklearner_train_time = zeros(1, length(weaklearner_list));
weaklearner_test_time = zeros(1, length(weaklearner_list));
weaklearner_best_test_result = struct();
weaklearner_worst_test_result = struct();

for i = 1:length(weaklearner_list)
    acc_list = [];
    train_time_list = [];
    test_time_list = [];
    test_results = {};
    fprintf("# weaklearner: %i\n", i);

    % Run experiment 10 times for each weaklearner
    for k = 1:num_repeats
        [accuracy, train_time, test_time, test_result] = RF_function(best_num_trees, best_depth_trees, 10, weaklearner_list{i}, false);
        acc_list(end+1) = accuracy;
        train_time_list(end+1) = train_time;
        test_time_list(end+1) = test_time;
        test_results{k} = test_result;
    end
    
    % Calculate and store averages
    weaklearner_accuracy(i) = mean(acc_list);
    weaklearner_train_time(i) = mean(train_time_list);
    weaklearner_test_time(i) = mean(test_time_list);
    
    % Save best and worst case test results
    [~, best_idx] = max(acc_list);
    [~, worst_idx] = min(acc_list);
    weaklearner_best_test_result.(['num_',num2str(i)]) = test_results{best_idx};
    weaklearner_worst_test_result.(['num_',num2str(i)]) = test_results{worst_idx};
end

results.weaklearner_accuracy = weaklearner_accuracy;
results.weaklearner_train_time = weaklearner_train_time;
results.weaklearner_test_time = weaklearner_test_time;
results.weaklearner_best_test_result = weaklearner_best_test_result;
results.weaklearner_worst_test_result = weaklearner_worst_test_result;
save('result.mat', 'results', '-append');  % Save after each experiment

% Plot for weaklearner
figure;
subplot(1,3,1);
bar(categorical(weaklearner_list), weaklearner_accuracy);
title('Accuracy for weaklearner');
xlabel('Weaklearner');
ylabel('Accuracy');

subplot(1,3,2);
bar(categorical(weaklearner_list), weaklearner_train_time);
title('Training Time for weaklearner');
xlabel('Weaklearner');
ylabel('Training Time (s)');

subplot(1,3,3);
bar(categorical(weaklearner_list), weaklearner_test_time);
title('Testing Time for weaklearner');
xlabel('Weaklearner');
ylabel('Testing Time (s)');
saveas(gcf, 'Graph3.png');

% Final save of results
save('result.mat', 'results');