% Want to extract:
vocab_size = [4,16,64,256];
class.label = [2,2,5];
class.sample = [2,4,3];

data_path = './Caltech_101';
class_list = dir(data_path);
class_list = {class_list(3:end).name}; % 10 classes name

figure;
    
% Plot histograms for each data sample (5 per class)
cnt = 1;
for j = 1:length(class.label)
    c = class.label(j);
    sub_path = fullfile(data_path,class_list{c});
    img_list = dir(fullfile(sub_path,'*.jpg'));
    I = imread(fullfile(sub_path,img_list(class.sample(j)).name));
                    
    if size(I,3) == 3
        I = rgb2gray(I);
    end
    
    subaxis(length(class.label), length(vocab_size)+1, cnt, 'SpacingVert', 0, 'MR', 0);
    imshow(I);
    drawnow;
    cnt = cnt + 1;
    
    for i = 1:length(vocab_size)
        hist_dim = vocab_size(i);
        load(sprintf("./experiment_data/kmeans_vocab(%d).mat", hist_dim));
        
        histograms = data_train(:, 1:end-1);  % Remove labels
        labels = data_train(:, end);          % Extract labels

        class_indices = 15 * (class.label(j)-1) + class.sample(j);

        % Extract the histogram for the i-th sample in the current class
        histogram = histograms(class_indices, :);
            
        % Plot each histogram in a subplot
        subplot(length(class.label), length(vocab_size)+1, cnt);  % Position of the current subplot
        bar(histogram);
            
        % Label each subplot
        title(sprintf('Class %d, Sample %d', class.label(j), class.sample(j)));
        xlabel('Histogram Bins');
        ylabel('Frequency');

        cnt = cnt + 1;
    end
end

%% 2. COVARIANCE MTX
for i = 1:length(vocab_size)
    histogram_list = [];
    for j = 1:length(class.label)
        hist_dim = vocab_size(i);
        load(sprintf("./experiment_data/kmeans_vocab(%d).mat", hist_dim));
        
        histograms = data_train(:, 1:end-1);  % Remove labels
        labels = data_train(:, end);          % Extract labels

        class_indices = 15 * (class.label(j)-1) + class.sample(j);

        % Extract the histogram for the i-th sample in the current class
        histogram = histograms(class_indices, :);
        histogram_list(j,:) = histogram;
    end
    disp(pdist2(histogram_list,histogram_list, 'cosine'))
    
    image_indices = ["Class2, Sample2", "Class2, Sample4", "Class5, Sample3"];

    similarity_matrix = 1.0 - pdist2(histogram_list, histogram_list, 'cosine');
    figure;
    imagesc(similarity_matrix); % Display matrix as an image
    colorbar; % Add color bar to indicate similarity scale
    title('Cosine Similarity Matrix');
    colormap('cool'); % Use a color map (e.g., 'hot', 'jet', etc.)
    caxis([0.0,1.0]);
    axis square; % Make axes square for better proportions

    % Add image index names to the axes
    xticks(1:length(image_indices)); % Set x-tick positions
    xticklabels(image_indices); % Assign x-tick labels
    yticks(1:length(image_indices)); % Set y-tick positions
    yticklabels(image_indices); % Assign y-tick labels

    % Display similarity values inside each cell
    textStrings = arrayfun(@(val) sprintf('%.4f', val), similarity_matrix, 'UniformOutput', false); % Format values to 4 decimal places
    [x, y] = meshgrid(1:length(image_indices), 1:length(image_indices)); % Grid for text placement
    text(x(:), y(:), textStrings, 'HorizontalAlignment', 'center', 'Color', 'black'); % Add text
    
end

