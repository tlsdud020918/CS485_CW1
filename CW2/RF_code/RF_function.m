function [train_accuracy, test_accuracy, train_time, test_time, train_result, test_result] = RF_function(num_trees, depth_trees, splitnum, weaklearner, conf_mtx)

hist_visualize = 0;

param.num = num_trees;
param.depth = depth_trees;    % trees depth
param.splitNum = splitnum; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = weaklearner;
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

[data_train, data_test] = getData('Caltech'); 
%disp(size(data_train))
%disp(size(data_test))

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
test_accuracy = accuracy_rf; % test accuracy

data_test_temp = data_test; % train accuracy
data_test = data_train;
testTrees_script;
train_result = c;
train_accuracy = accuracy_rf;
data_test = data_test_temp;

% 2. confusioin matrix

% 3. example success/failures ??? -> 구상 필요

% 4. histogram visualization
if hist_visualize == 1
    histograms = data_train(:, 1:end-1);  % Remove labels
    labels = data_train(:, end);          % Extract labels
    
    % Number of classes and histogram dimensions
    num_classes = 5;
    hist_dim = size(histograms, 2);
    
    figure;
    
    % Plot histograms for each data sample (5 per class)
    for c = 1:num_classes
        % Find the indices of the current class
        class_indices = find(labels == c);
        
        for i = 1:5  % Plot first 5 samples per class
            % Extract the histogram for the i-th sample in the current class
            histogram = histograms(class_indices(i), :);
            
            % Plot each histogram in a subplot
            subplot(num_classes, 5, (c-1)*5 + i);  % Position of the current subplot
            bar(histogram);
            
            % Label each subplot
            title(sprintf('Class %d, Sample %d', c, i));
            xlabel('Histogram Bins');
            ylabel('Frequency');
        end
    end
    
    % Adjust layout for better spacing
    sgtitle('Histograms of Samples by Class');  % Add overall title
end


