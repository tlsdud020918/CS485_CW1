function [histograms, elapsed_time] = q1_quantization_kmeans(descriptors, vocabulary)

num_class = 10;
num_img = 15;
vocab_size = size(vocabulary, 2);

% Preallocate histogram
histograms = zeros(num_class * num_img, vocab_size);

% Start timing
tic;

% Combine descriptors for all images into a single matrix
for i = 1:num_class
    for j = 1:num_img
        % Extract descriptors for image (i, j)
        desc_ij = single(descriptors{i, j});  % Convert to double if necessary
        
        % Compute pairwise distances between descriptors and vocabulary centers
        distances = pdist2(desc_ij', vocabulary');  % Shape: [num_desc, vocab_size]
        
        % Find nearest clusters for all descriptors
        [~, cluster_indices] = min(distances, [], 2); % Assign each descriptor to the nearest cluster
        
        % Update histogram for the current image
        histogram = histcounts(cluster_indices, 1:vocab_size+1); % 1xK
            
        % Normalize the histogram
        histogram = histogram / sum(histogram);

        linear_idx = (i - 1) * num_img + j;  % Linear index in the histogram
        histograms(linear_idx, :) = histogram;
    end
end

elapsed_time = toc;  % Record elapsed time
end
