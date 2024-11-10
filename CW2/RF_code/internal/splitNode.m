function [node,nodeL,nodeR] = splitNode(data,node,param, visualise)
% Split node

%visualise = 1;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    
    % Split function - Modify here and try other types of split function
    switch param.weaklearner
        case 'axis-aligned'
            dim = randi(D-1); % Pick one random dimension
            d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
            d_max = single(max(data(:,dim))) - eps;
            t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
            idx_ = data(:,dim) < t;
        case 'two-pixel'
            dim = randperm(D-1, 2); % Pick two random dimension
            idxi = dim(1);
            idxj = dim(2);
            
            xi = data(:, idxi);
            xj = data(:, idxj);
            
            dist = xi - xj;
            
            d_min = single(min(dist)) + eps;
            d_max = single(max(dist)) - eps;
            
            t = d_min + rand*((d_max - d_min));
            idx_ = dist < t;
        case 'linear'
            cond = true;
            while cond
                dim = randperm(D-1, 2);
                t = randn(3, 1);
                phi = cat(2, data(:, dim), ones(N, 1));
                idx_ = (double(phi) * t) < 0;
                cond = sum(idx_) == 0 || sum(~idx_) == 0;
            end
        case 'non-linear'
            cond = true;
            while cond
                dim = randperm(D-1, 2);
                t = randn(3, 3);
                idx_ = true(N,1);
                for i = 1:N
                    phi = cat(2, data(i, dim), ones(1, 1)); % N * 3
                    phi = double(phi);
                    idx_(i) = (phi * t * phi.') < 0;
                end
                cond = sum(idx_) == 0 || sum(~idx_) == 0;
            end
    end
    
    ig = getIG(data,idx_); % Calculate information gain
    
    %if visualise
    %    visualise_splitfunc(idx_,data,dim,t,ig,n);
    %    pause();
    %end
    
    if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
    end
    
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end