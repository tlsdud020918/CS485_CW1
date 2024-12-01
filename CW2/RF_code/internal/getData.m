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
vocab_size = 256;
is_kmeans = 0;

showImg = 0;

max_descriptor = 1e5;
num_class = 10;
num_img = 15;

% SIFT variables
PHOW_STEP = 8; % The lower the denser. Select from {2,4,8,16}
%PHOW_SIZES = [4 6 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_SIZES = [4 8 10];

% saving variables
feature_save = 1;
vocab_save = 1;
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
        data_path = './Caltech_101';
        class_list = dir(data_path);
        class_list = {class_list(3:end).name}; % 10 classes name

        % 1. load image & extract SIFT descriptors
        disp("Load train images...")
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end

        imgIdx_tr = 1:15;
        cnt = 1;
        for c = 1:length(class_list)
            sub_path = fullfile(data_path,class_list{c});
            img_list = dir(fullfile(sub_path,'*.jpg'));
           
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(sub_path,img_list(imgIdx_tr(i)).name));
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end

                % Visualise
                if i <= 5 && showImg
                    subaxis(length(class_list),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt + 1;
                    drawnow;
                end

                I = single(I) / 255; % scaling into [0,1] -> vl_phow의 document 참조
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(I, 'Step', PHOW_STEP, 'Sizes', PHOW_SIZES); %  extracts PHOW features (multi-scaled Dense SIFT)
                desc_tr_labeled{c,i} = [desc_tr{c,i}; repmat(c, 1, size(desc_tr{c,i}, 2))];
                %[rows, cols] = size(desc_tr{c, i});
                %fprintf('Descriptor shape for class %d, image %d: %d x %d\n', c, i, rows, cols);
            end
        end
        fprintf('Total %d descriptors are extracted\n', size(cat(2,desc_tr{:}), 2));
        % end: 1. load image & extract SIFT descriptors
        
        % 2. Randomly select 100k SIFT descriptors (before clustering)
        desc_selected_labeled = single(vl_colsubset(cat(2,desc_tr_labeled{:}), max_descriptor));
        desc_selected = desc_selected_labeled(1:end-1, :);
        % end: 2. Randomly select 100k SIFT descriptors (before clustering)

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
            [histograms, hist_time] = q1_quantization_kmeans(desc_tr, kmeans_vocab);
        end
        if is_kmeans == 0
            disp("RF trainset histogram building...");
            [histograms, hist_time] = q3_quantization_rf(desc_tr, vocab_trees, vocab_size, weaklearner);
        end

        % X = (이미지 개수 * vocab_size), Y = (이미지 개수 * 1)
        X = histograms;
        Y = repelem(1:10, 15)';
        % end: 4. Build histograms of train data (vector quantization)
        
        % 5. Return output
        data_train = [X Y];
        % end: 5. Return output

        % desc_selected = 100k selected SIFT descriptors
        % centers = K-means clustering center
        if feature_save == 1
            save descriptor.mat desc_selected
        end
        if vocab_save == 1
            if is_kmeans == 1
                save kmeans_vocab.mat kmeans_vocab
            end
        end
        % saving end
  
end

switch MODE
    case 'Caltech' % Caltech101: testing data extraction
        data_path = './Caltech_101';
        imgIdx_te = 16:30;
        
        % 1. load test image & extract SIFT descriptors
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Test image samples');
        end

        disp('Processing testing images...');
        cnt = 1;
        for c = 1:length(class_list)
            sub_path = fullfile(data_path,class_list{c});
            img_list = dir(fullfile(sub_path,'*.jpg'));
           
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(sub_path,img_list(imgIdx_te(i)).name));
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end

                % Visualise
                if i <= 5 && showImg
                    subaxis(length(class_list),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt + 1;
                    drawnow;
                end

                I = single(I) / 255; % scaling into [0,1] -> vl_phow의 document 참조
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_te{c,i}] = vl_phow(I, 'Step', PHOW_STEP, 'Sizes', PHOW_SIZES); %  extracts PHOW features (multi-scaled Dense SIFT)
                %[rows, cols] = size(desc_tr{c, i});
                %fprintf('Descriptor shape for class %d, image %d: %d x %d\n', c, i, rows, cols);
            end
        end
        fprintf('Total %d descriptors are extracted\n', size(cat(2,desc_te{:}), 2));
        % end: 1. load test image & extract SIFT descriptors

        % 2. Build histograms of test data (vector quantization)
        if is_kmeans == 1
            disp("k-means testset histogram building...")
            [histograms, hist_time] = q1_quantization_kmeans(desc_te, kmeans_vocab);
        end
        if is_kmeans == 0
            disp("RF testset histogram building...")
            [histograms, hist_time] = q3_quantization_rf(desc_tr, vocab_trees, vocab_size, weaklearner);
        end

        % X = (이미지 개수 * vocab_size), Y = (이미지 개수 * 1)
        X = histograms;
        Y = repelem(1:10, 15)';
        % end: 2. Build histograms of test data (vector quantization)

        if is_kmeans == 1
            disp("k-means testset histogram building...")
        end
        if is_kmeans == 0
            % TODO!!!!!
            disp("RF testset histogram building...")
        end
        
        data_query = [X Y];
        
    otherwise
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end