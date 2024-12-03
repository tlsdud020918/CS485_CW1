function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle

% TODO: obtain the visual vocabulary and the bag-of-words histograms for both training and testing data.
% train: visual vocab 추출 + vector quantization으로 각 이미지에 대한 histogram 추출
% test: vector quantization으로 각 이미지에 대한 histogram 추출
% VARIABLES START
global vocab_size is_kmeans showImg feature_save vocab_save;
global vocab_time train_hist_time test_hist_time vocab_param;

max_descriptor = 1e5;
num_class = 10;
num_img = 15;

% VARIABLES END

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4;
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8;
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];

    case 'Caltech' % Caltech101: training data extraction
        if is_kmeans == 1 && vocab_save == 0
            filename = sprintf('./experiment_data/kmeans_vocab(%d).mat', vocab_size);
            load(filename)
            return
        end

        if feature_save == 1
            % 1. load image & extract SIFT descriptors
            [desc_tr, desc_tr_labeled] = q1_ImgtoFeature(1, showImg);
            % end: 1. load image & extract SIFT descriptors
            
            % 2. Randomly select 100k SIFT descriptors (before clustering)
            desc_selected_labeled = single(vl_colsubset(cat(2,desc_tr_labeled{:}), max_descriptor));
            desc_selected = desc_selected_labeled(1:end-1, :);
            % end: 2. Randomly select 100k SIFT descriptors (before clustering)
        
        else
            load descriptor.mat
        end

        % 3. Build visual codebook: Kmeans / RF method
        if is_kmeans == 1
            disp("k-means codebook building...")
            [kmeans_vocab, assignments, vocab_time] = q1_codebook_kmeans(desc_selected, vocab_size);
            fprintf("k-means codebook building time: %f sec \n", vocab_time)
        end
        if is_kmeans == 0
            disp("RF codebook building...")
            % change vocab_size into real result's dimension
            [vocab_trees, vocab_time, vocab_size, weaklearner] = q3_codebook_rf(desc_selected_labeled', vocab_size);
            fprintf("RF codebook building time: %f sec \n", vocab_time)
        end
        % end: 3. Build visual codebook: Kmeans / RF method
        
        % 4. Build histograms of train data (vector quantization)
        if is_kmeans == 1
            disp("k-means trainset histogram building...");
            [histograms, train_hist_time] = q1_quantization_kmeans(desc_tr, kmeans_vocab);
        end
        if is_kmeans == 0
            disp("RF trainset histogram building...");
            [histograms, train_hist_time] = q3_quantization_rf(desc_tr, vocab_trees, vocab_size, weaklearner);
        end

        % X = (이미지 개수 * vocab_size), Y = (이미지 개수 * 1)
        X = histograms;
        Y = repelem(1:10, 15)';
        % end: 4. Build histograms of train data (vector quantization)
        
        % 5. Return output
        data_train = [X Y];
        % end: 5. Return output

        % saving end
  
end

switch MODE
    case 'Caltech' % Caltech101: testing data extraction
        % 1. load test image & extract SIFT descriptors
        if feature_save == 1
            [desc_te, ~] = q1_ImgtoFeature(0, showImg);
        end
        % end: 1. load test image & extract SIFT descriptors

        if feature_save == 1
            % Feature extraction result saving
            save descriptor.mat desc_selected desc_selected_labeled desc_tr desc_te
        end

        % 2. Build histograms of test data (vector quantization)
        if is_kmeans == 1
            disp("k-means testset histogram building...")
            [histograms, test_hist_time] = q1_quantization_kmeans(desc_te, kmeans_vocab);
        end
        if is_kmeans == 0
            disp("RF testset histogram building...")
            [histograms, test_hist_time] = q3_quantization_rf(desc_te, vocab_trees, vocab_size, weaklearner);
        end

        % X = (이미지 개수 * vocab_size), Y = (이미지 개수 * 1)
        X = histograms;
        Y = repelem(1:10, 15)';
        % end: 2. Build histograms of test data (vector quantization)

        data_query = [X Y];

        % codebook & histogram save
        if is_kmeans == 1
            filename = sprintf('./experiment_data/kmeans_vocab(%d).mat', vocab_size);
            if vocab_save == 1
                save(filename, "kmeans_vocab", "data_train", "data_query")
            end
        end
        
    otherwise
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end