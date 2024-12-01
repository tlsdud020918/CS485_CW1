function [trees, elapsed_time, RF_vocab_size, weaklearner] = q3_codebook_rf(descriptors, vocab_size)

% VARIABLES START
is_train = 1;

vocab_param.num = 32;
vocab_param.depth = log2(vocab_size / vocab_param.num) + 1;    % trees depth
vocab_param.splitNum = 3; % Number of trials in split function
vocab_param.split = 'IG'; % Currently support 'information gain' only
vocab_param.weaklearner = 'axis-aligned';
weaklearner = vocab_param.weaklearner;
% weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'
% VARIABLES END

if is_train == 0
    load question3.mat
    assignments = [];
    elapsed_time = 0;
    return
end

tic;
trees = growTrees(descriptors, vocab_param);
RF_vocab_size = trees(vocab_param.num).leaf(end).label(1);
fprintf("Result vocabulary dimension: %d\n", RF_vocab_size);
elapsed_time = toc;