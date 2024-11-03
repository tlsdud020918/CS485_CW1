%% 3. Experiment with weaklearner using best cases from previous steps
init ;
num_repeats = 1;  % Increased for better accuracy estimation
best_num_trees = 250;
depth_trees_list = [1,3,5,7,10];  % Different depths to test
weaklearner_list = {'axis-aligned', 'two-pixel', 'linear'};
num_weaklearners = length(weaklearner_list);

% Preallocate matrices to store results
weaklearner_accuracy = zeros(num_weaklearners, length(depth_trees_list));
weaklearner_train_time = zeros(num_weaklearners, length(depth_trees_list));
weaklearner_test_time = zeros(num_weaklearners, length(depth_trees_list));

% Loop through each weak learner and depth
for i = 1:num_weaklearners
    for j = 1:length(depth_trees_list)
        acc_list = [];
        train_time_list = [];
        test_time_list = [];
        test_results = {};
        fprintf("# Weaklearner: %s, Depth: %d\n", weaklearner_list{i}, depth_trees_list(j));

        % Run experiment for each configuration
        for k = 1:num_repeats
            [accuracy, train_time, test_time, test_result] = RF_function(best_num_trees, depth_trees_list(j), 10, weaklearner_list{i}, false);
            acc_list(end + 1) = accuracy;
            train_time_list(end + 1) = train_time;
            test_time_list(end + 1) = test_time;
            test_results{k} = test_result;
        end
        
        % Calculate and store averages
        weaklearner_accuracy(i, j) = mean(acc_list);
        weaklearner_train_time(i, j) = mean(train_time_list);
        weaklearner_test_time(i, j) = mean(test_time_list);
    end
end

% Save results in the results structure
results.weaklearner_accuracy = weaklearner_accuracy;
results.weaklearner_train_time = weaklearner_train_time;
results.weaklearner_test_time = weaklearner_test_time;
save('result.mat', 'results', '-append');  % Save after each experiment

% Plot for weaklearner
figure;
hold on;

% Plot accuracy for each weak learner
for i = 1:num_weaklearners
    plot(depth_trees_list, weaklearner_accuracy(i, :), '-o', 'DisplayName', weaklearner_list{i});
end
title('Accuracy for Weak Learner');
xlabel('Tree Depth');
ylabel('Accuracy');
legend show;  % Show legend for weak learners
grid on;

% Save the figure
saveas(gcf, 'Graph3_accuracy.png');

% Plot training time for each weak learner
figure;
hold on;
for i = 1:num_weaklearners
    plot(depth_trees_list, weaklearner_train_time(i, :), '-o', 'DisplayName', weaklearner_list{i});
end
title('Training Time for Weak Learner');
xlabel('Tree Depth');
ylabel('Training Time (s)');
legend show;
grid on;

% Save the figure
saveas(gcf, 'Graph3_train_time.png');

% Plot testing time for each weak learner
figure;
hold on;
for i = 1:num_weaklearners
    plot(depth_trees_list, weaklearner_test_time(i, :), '-o', 'DisplayName', weaklearner_list{i});
end
title('Testing Time for Weak Learner');
xlabel('Tree Depth');
ylabel('Testing Time (s)');
legend show;
grid on;

% Save the figure
saveas(gcf, 'Graph3_test_time.png');

% Final save of results
save('result_weaklearners.mat', 'results');
