function label = testTrees_fast(data,tree, weaklearner)
% Faster version - pass all data at same time
cnt = 1;

if nargin < 3
    weaklearner = 'axis-aligned';
end

for T = 1:length(tree)
    idx{1} = 1:size(data,1);
    for n = 1:length(tree(T).node);
        if ~tree(T).node(n).dim
            leaf_idx = tree(T).node(n).leaf_idx;
            if ~isempty(tree(T).leaf(leaf_idx))
                label(idx{n}',T) = tree(T).leaf(leaf_idx).label;
            end
            continue;
        end

        switch weaklearner
            case 'axis-aligned'
                idx_left = data(idx{n},tree(T).node(n).dim) < tree(T).node(n).t;
            case 'two-pixel'
                idx_left = data(idx{n},tree(T).node(n).dim(1)) - data(idx{n},tree(T).node(n).dim(2)) < tree(T).node(n).t;
            case 'linear'
                t = tree(T).node(n).t; % 3*1 format
                dim = tree(T).node(n).dim;
                [N,~] = size(data(idx{n}, :));
                phi = cat(2, data(idx{n},dim), ones(N,1));
                idx_left = (double(phi) * t) < 0;
            case 'non-linear'
                t = tree(T).node(n).t; % 3*3 format
                dim = tree(T).node(n).dim; % 2 value
                [N,~] = size(data(idx{n}, :));
                idx_left = true(N,1);
                for i = 1:N
                    phi = cat(2, data(i, dim), ones(1, 1)); % N * 3
                    phi = double(phi);
                    idx_left(i) = (phi * t * phi.') < 0;
                end
        end
        idx{n*2} = idx{n}(idx_left');
        idx{n*2+1} = idx{n}(~idx_left');
    end
end

end

