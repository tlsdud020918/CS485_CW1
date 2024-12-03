clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

param.num = 250;
param.depth = 10;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.weaklearner = 'axis-aligned';
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

% data load
global vocab_size is_kmeans showImg feature_save vocab_save;
vocab_size = 1024;
is_kmeans = 1;
showImg = 0;
feature_save = 0;
vocab_save = 0;

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

% Create a figure for the histograms
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