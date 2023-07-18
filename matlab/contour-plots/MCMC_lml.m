function MCMC_lml(ID, dataset, l_min, l_max)
%% MCMC_lml - Run MCMC lml estimation

    % Add GPML to the path
    addpath /path/to/GPML
    startup
   
    % Parse input to indices
    i = idivide(int32(ID), 21) + 1;
    j = mod(int32(ID), 21) + 1;


%% Load data

    % Load data
    data = load(['data/' dataset '.mat']);
    
    % Extract training/test input-output pairs
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
    likfunc = @likErf; hyp.lik = [];
        
    
%% Compute the marginal likelihood on training set
    
    % Sampler parameters
    par.sampler = 'ess';
    par.Nsample = 8000;
    par.Nskip = 1000;
    par.Nburnin = 5000;
    par.Nais = 3;

    % Use AIS for computing an estimate of the NLL by directly passing
    % the options to infMCMC instead of doing the default
    % [nlZ,dnlZ] = gp(hyp, @infMCMC, meanfunc, covfunc, likfunc, train_X, train_Y);
    [post,nlZ] = infMCMC(hyp, {meanfunc}, covfunc, {likfunc}, train_X, train_Y, par);
    lml = -nlZ/N;

    % Report and save
    text = ['i = ', num2str(i), ' j = ', num2str(j), ' log marginal = ', num2str(lml)];
    disp(text);
    mkdir lml
    save(join(['./lml/', dataset, '/', 'i=',num2str(i),'j=',num2str(j),'.mat']),'lml')













