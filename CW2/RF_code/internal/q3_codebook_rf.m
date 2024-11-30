function [trees, elapsed_time] = q3_codebook_rf(descriptors, vocab_size)

% VARIABLES
is_train = 1;

vocab_param.num = 100;
vocab_param.depth = log2(vocab_size) + 1;    % trees depth
vocab_param.splitNum = 3; % Number of trials in split function
vocab_param.split = 'IG'; % Currently support 'information gain' only
vocab_param.weaklearner = 'axis-aligned';
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

if is_train == 0
    load question3.mat
    assignments = [];
    elapsed_time = 0;
    return
end

tic;
trees = growTrees(descriptors, vocab_param);
elapsed_time = toc;