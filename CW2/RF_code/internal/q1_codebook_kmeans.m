function [centers, assignments, time] = q1_codebook_kmeans(descriptors, vocab_size)

tic;
[centers, assignments] = vl_kmeans(descriptors, vocab_size, 'distance', 'l2');
time = toc;