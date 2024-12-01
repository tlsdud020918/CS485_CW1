function [histograms, elapsed_time] = q3_quantization_rf(descriptors, vocab_trees, vocab_size, weaklearner)

num_class = 10;
num_img = 15;

% Preallocate histogram
histograms = zeros(num_class * num_img, vocab_size);

% Start timing
tic;

% Combine descriptors for all images into a single matrix
for i = 1:num_class
    for j = 1:num_img
        % Extract descriptors for image (i, j)
        desc_ij = single(descriptors{i, j});  % Convert to double if necessary
        
        leaf_assign = testTrees_fast(desc_ij', vocab_trees, weaklearner);
        
        % Flatten the leaf indices across all trees
        leaf_indices = leaf_assign(:); % Flatten to 1D vector
        
        % Create a histogram of the leaf indices
        histogram = histcounts(leaf_indices, 1:(vocab_size + 1)); % 1 x vocab_size
        
        % Normalize the histogram to sum to 1
        histogram = histogram / sum(histogram);

        linear_idx = (i - 1) * num_img + j;  % Linear index in the histogram
        histograms(linear_idx, :) = histogram;
    end
end

elapsed_time = toc;  % Record elapsed time
end
