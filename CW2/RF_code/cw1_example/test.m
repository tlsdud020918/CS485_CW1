init ;

%RF_function(double(100), double(5), double(3), 'axis-aligned', 0)

for N = [1,3,5,10,20] % Number of trees, try {1,3,5,10, or 20}
    param.num = N;
    param.depth = 5;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only
    param.weaklearner = 'non-linear';
    % weaklearner: 'axis-aligned', 'two-pixel', 'linear', 'non-linear'

    % Select dataset
    [data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
    
    % Train Random Forest
    trees = growTrees(data_train, param);
    
    % Test Random Forest
    testTrees_script;
    
    % Visualise
    visualise(data_train,p_rf,[],0);
    disp('4.1: Press any key to continue');
    pause;
end