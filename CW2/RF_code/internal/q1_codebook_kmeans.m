function [centers, assignments, time] = q1_codebook_kmeans(descriptors, vocab_size)

% VARIABLES
is_train = 1;

if is_train == 0
    load question1.mat
    assignments = [];
    time = 0;
    return
end

tic;
[centers, assignments] = vl_kmeans(descriptors, vocab_size, 'distance', 'l2');
time = toc;