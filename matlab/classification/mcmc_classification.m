function mcmc_classification(dataset_id, seed)
%% mcmc_classification - Run GP classification with MCMC

  % Add path to GPstuff
  addpath /path/to/gpstuff
  
  % Run GPstuff startup scripts
  startup
  

%% Retrieve dataset name

  dataset_list = load('dataset_list.mat');
  dataset = getfield(dataset_list,sprintf('dataset%i',dataset_id));


%% Model specification

  % Construct the priors for the parameters of covariance functions
  pl = prior_logunif();
  pm = prior_logunif();
  
  % Store results
  LA_lpd = nan(5,1);
  EP_lpd = nan(5,1);
  MCMC_lpd = nan(5,1);
  LA_acc = nan(5,1);
  EP_acc = nan(5,1);  
  MCMC_acc = nan(5,1);  
  
  % Loda dateset with folds
  allfolds = load(sprintf('data/%s%d.mat',dataset, seed));
  
  % Run 5-fold cv
  for fold = 1:5
  
    % Load data
    data = getfield(allfolds,sprintf('fold%i',fold-1)); %#ok

    % Pick data
    x = data.train_X;
    y = (data.train_Y<.5)*2-1;
    xt = data.test_X;
    yt = (data.test_Y<.5)*2-1;
    
    % GP covariance function
    gpcf = gpcf_matern52('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
        
    % Likelihood
    lik = lik_probit();
    
    % *** Laplace ***
    
    % Create the GP structure
    gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6, ...
            'latent_method', 'Laplace');
        
    % Set the options for the optimization
    opt = optimset('TolFun',1e-3,'TolX',1e-3,'display','iter');
    
    % Optimize with the scaled conjugate gradient method
    gp = gp_optim(gp,x,y,'opt',opt);
    
    % Predictions to test points
    [Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, xt, 'yt', yt);
    
    % Mean LPD
    LA_lpd(fold) = mean(lpyt);

    % Accuracy
    LA_acc(fold) = mean((Eft>0)==(yt>.5));    
    
    % *** EP ***    
    
    % Create the GP structure
    gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6, ...
            'latent_method', 'EP');
    
    % Set the options for the optimization
    opt = optimset('TolFun',1e-3,'TolX',1e-3,'display','iter');
    
    % Optimize with the scaled conjugate gradient method
    gp = gp_optim(gp,x,y,'opt',opt);
    
    % Predictions to test points
    [Eft, Varft, lpyt] = gp_pred(gp, x, y, xt, 'yt', yt);
    
    % Mean LPD
    EP_lpd(fold) = mean(lpyt);

    % Accuracy
    EP_acc(fold) = mean((Eft>0)==(yt>.5));
    
    % *** MCMC ***

    % Create the GP structure
    gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6, ...
            'latent_method', 'MCMC');

    % Note that MCMC for latent values requires often more jitter
    gp = gp_set(gp, 'latent_method', 'MCMC', 'jitterSigma2', 1e-6);

    % Sample using default method, that is, surrogate and elliptical slice
    % samplers these samplers are quite robust with default options
    [gp_rec,g,opt]=gp_mc(gp, x, y, 'nsamples', 10200, 'display', 100);

    % Remove burn-in and thin
    gp_rec=thin(gp_rec,201,2);

    % Make predictions
    [Ef_mc, Varf_mc, lpy_mc, Ey_mc, Vary_mc] = ...
       gp_pred(gp_rec, x, y, xt, 'yt', yt);

    % Mean LPD
    MCMC_lpd(fold) = mean(lpy_mc);

    % Accuracy
    MCMC_acc(fold) = mean((Ef_mc>0)==(yt>.5));
        
  end


%% Show and store

  % Table
  fprintf('%10s  :      --LA--            --EP--            --MCMC--\n','')
  fprintf('%10s  : %.3f \\pm %.3f  %.3f \\pm %.3f  %.3f \\pm %.3f \n\n',dataset, ...
      mean(LA_lpd),std(LA_lpd), ...
      mean(EP_lpd),std(EP_lpd), ...
      mean(MCMC_lpd),std(MCMC_lpd))
  
  % Save results
  mkdir('result')  
  save(sprintf('result/mcmc-%s%d.mat',dataset, seed), ...
      'LA_lpd','EP_lpd','LA_acc','EP_acc','MCMC_lpd','MCMC_acc')
  
