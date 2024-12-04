function [descriptors, descriptors_labeled] = q1_ImgtoFeature(istrain, showImg)

% SIFT variables
PHOW_STEP = 8; % The lower the denser. Select from {2,4,8,16}
PHOW_SIZES = [4 6 8 10]; % Multi-resolution, these values determine the scale of each layer.
% SIFT variables end

% 1. load image & extract SIFT descriptors
if istrain == 1
    message = "Train";
else
    message = "Test";
end
fprintf("Load %s images... \n", message);

if showImg
    figure('Units','normalized','Position',[.05 .1 .4 .9]);
    titleText = sprintf("%s image samples", message);
    suptitle(titleText);
end
    
imgIdx_tr = 1:15;
imgIdx_te = 16:30;

if istrain == 1
    imgIdx = imgIdx_tr;
else
    imgIdx = imgIdx_te;
end
cnt = 1;

data_path = './Caltech_101';
class_list = dir(data_path);
class_list = {class_list(3:end).name}; % 10 classes name
for c = 1:length(class_list)
     sub_path = fullfile(data_path,class_list{c});
     img_list = dir(fullfile(sub_path,'*.jpg'));
               
     for i = 1:length(imgIdx)
        I = imread(fullfile(sub_path,img_list(imgIdx(i)).name));
                    
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
         [~, descriptors{c,i}] = vl_phow(I, 'Step', PHOW_STEP, 'Sizes', PHOW_SIZES); %  extracts PHOW features (multi-scaled Dense SIFT)
         descriptors_labeled{c,i} = [descriptors{c,i}; repmat(c, 1, size(descriptors{c,i}, 2))];
         %[rows, cols] = size(desc_tr{c, i});
         %fprintf('Descriptor shape for class %d, image %d: %d x %d\n', c, i, rows, cols);
    end
end
fprintf('Total %d descriptors are extracted \n', size(cat(2,descriptors{:}), 2));
% end: 1. load image & extract SIFT descriptors