clear all; close all; 
% Initialisation
init; clc;
run("vlfeat-0.9.21/toolbox/vl_setup")

% Experiment: K value
n = 1:9; % Define the range of exponents (you can adjust the range as needed)
experiment_array = 2.^n; % Compute powers of 2
%repeatnum = 5;
repeatnum = 1;

% K-means clustering VARIABLES
global vocab_size is_kmeans showImg feature_save vocab_save;
global vocab_time train_hist_time test_hist_time;
vocab_size = 256;
is_kmeans = 1;
showImg = 0;
feature_save = 0;
vocab_save = 1;
% end: K-means clustering VARIABLES

% RF classification VARIABLES
num_trees = 100;
depth_trees= 5;    % trees depth
splitnum = 3; % Number of trials in split function
weaklearner = 'axis-aligned';
conf_mtx = 0;
% end: RF classification VARIABLES

vocabT_list = zeros(length(experiment_array),1);
train_histT_list = zeros(length(experiment_array),1);
test_histT_list = zeros(length(experiment_array),1);
for temp=1:repeatnum
    for cnt=1:length(experiment_array)
        vocab_size = experiment_array(cnt);
        [train_accuracy, test_accuracy, train_time, test_time, ~,~] = RF_function(num_trees, depth_trees, splitnum, weaklearner, conf_mtx);
        vocabT_list(cnt) = vocabT_list(cnt) + vocab_time;
        train_histT_list(cnt) = train_histT_list(cnt) + train_hist_time;
        test_histT_list(cnt) = test_histT_list(cnt) + test_hist_time;
    end
end

vocabT_list = vocabT_list / repeatnum;
train_histT_list = train_histT_list / repeatnum;
test_histT_list = test_histT_list / repeatnum;
% save("question1.mat", "vocabT_list", "train_histT_list", "test_histT_list");