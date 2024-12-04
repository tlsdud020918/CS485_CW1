hist_dim = 64;
isTrain = 1;
load(sprintf("./experiment_data/RF_vocab(%d).mat", hist_dim));
run("vlfeat-0.9.21/toolbox/vl_setup")

if isTrain == 1
    histograms = data_train(:, 1:end-1);  % Remove labels
    labels = data_train(:, end);          % Extract labels
else
    histograms = data_query(:, 1:end-1);  % Remove labels
    labels = data_query(:, end);          % Extract labels
end

% Number of classes and histogram dimensions
num_classes = 10;

%showImg = 1;
%[~, ~] = q1_ImgtoFeature(0, showImg);
    
figure; % Create a new figure

% Find the maximum number of bins in histograms
num_bins = size(histograms, 2);

% Initialize a matrix to hold all histograms
all_histograms = NaN(num_classes, 5, num_bins); % Classes x Samples x Histogram Bins

% Organize histograms into a matrix
for c = 1:num_classes
    % Find indices of samples for this class
    class_indices = find(labels == c);
    for i = 1:min(5, length(class_indices)) % Limit to 5 samples per class
        all_histograms(c, i, :) = histograms(class_indices(i), :);
    end
end

%% Visualize histograms
for c = 1:num_classes
    for i = 1:5
        % Extract the histogram for class `c`, sample `i`
        histogram = squeeze(all_histograms(c, i, :));
        if ~all(isnan(histogram)) % Check if there is valid data
            % Define subplot grid
            subplot(num_classes, 5, (c-1)*5 + i);
            
            % Plot histogram
            bar(histogram, 'FaceColor', [0.2, 0.6, 0.8]); % Uniform color
            
            % Remove subplot details for compactness
            %axis tight; % Fit axis tightly around the bars
            xticks([]);
            
            % Add class labels on the leftmost plots
            if i == 1
                ylabel(sprintf('Class %d', c), 'FontSize', 10);
            end
            
            % Add sample labels on the bottom row
            if c == num_classes
                xlabel(sprintf('Sample %d', i), 'FontSize', 10);
            end
        end
    end
end

% Add an overarching title
if isTrain == 0
    sgtitle(sprintf('Test data histograms, vocabulary size=%d', hist_dim), 'FontSize', 14);
else
    sgtitle(sprintf('Train data histograms, vocabulary size=%d', hist_dim), 'FontSize', 14);
end