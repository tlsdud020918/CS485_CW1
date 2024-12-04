global vocab_size is_kmeans showImg feature_save vocab_save vocab_param;
global vocab_time train_hist_time test_hist_time;
vocab_size = 256;
is_kmeans = 0;
showImg = 0;
feature_save = 0;
vocab_save = 1;
vocab_param.num = 4; % prior experiment: 전체 256에 이거 32
vocab_param.depth = 5;    % trees depth
vocab_param.splitNum = 10; % Number of trials in split function
vocab_param.split = 'IG'; % Currently support 'information gain' only
vocab_param.weaklearner = 'axis-aligned';

[ data_train, data_query ] = getData('Caltech');
save('./experiment_data/RF_vocab(64).mat', 'data_train', 'data_query');