function MCMC_lp(ID, dataset, l_min, l_max)
%% MCMC_lp - Run MCMC for predictive density estimation

    % Add GPML to the path
    addpath /path/to/GPML
    startup
    
    % Parse input to indices
    i = idivide(int32(ID), 21) + 1;
    j = mod(int32(ID), 21) + 1;


%% Load data

    % Load data
    data = load(['data/' dataset '.mat']);
    
    % Extract training input-output pairs
    train_X = data.train_X;
    train_Y = data.train_Y;
    test_X = data.test_X;
    test_Y = data.test_Y;

    % Dimensionality
    [N, D] = size(train_X);
    

%% Experiment setup
    
    % Set the grid range
    log_l = linspace(l_min, l_max, 21);
    log_sigma = linspace(-1, 5, 21);

    % Define the GP model
    covfunc = {@covMaterniso, 5}; hyp.cov = [log_l(j), log_sigma(22-i)];
    meanfunc = @meanConst; hyp.mean = 0;
    likfunc = @likErf;
    
    
%% Compute the predictive density
        
    % Sampler parameters
    par.sampler = 'ess';
    par.Nsample = 8000;
    par.Nskip = 1000;
    par.Nburnin = 5000;
    par.Nais = 3;

    % Use infMCMC directly to pass the options
    [post,nlZ] = infMCMC(hyp, {meanfunc}, covfunc, {likfunc}, train_X, train_Y, par);
    [ymu,ys2,fmu,fs2,lp] = gp(hyp, @infMCMC, meanfunc, covfunc, likfunc, train_X, post, test_X, test_Y);

    % Mean predictive density (to normalize scale)    
    lp = mean(lp);

    % report and save
    text = ['i = ', num2str(i), ' j = ', num2str(j), ' log predictive = ', num2str(lp)];
    disp(text);
    mkdir('lp')
    save(join(['./lp/', dataset, '/', 'i=',num2str(i),'j=',num2str(j),'.mat']),'lp');


