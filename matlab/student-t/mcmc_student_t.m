function mcmc_student_t(dataset_id)
%% mcmc_student_t - Run GP benchmarks with Student-t likelihood

  % Add path to GPstuff
  addpath gpstuff
  
  % Run GPstuff startup scripts
  startup

  % Choose data set name to run
  if dataset_id == 1
        dataset = 'odata';
  elseif dataset_id == 2
        dataset = 'boston';
  else
        dataset = 'stock';
  end

  % Construct the priors for the parameters of covariance functions
  pl = prior_logunif();
  pm = prior_logunif();
  pn = prior_logunif();
  
  % Store results
  LA_lpd = nan(5,1);
  EP_lpd = nan(5,1);
  MCMC_lpd = nan(5,1);
  
  % Run 5-fold cv
  for fold = 1:5

    % Load data
    data = load(sprintf('data/%s_%i.mat',dataset,fold-1));
    
    % Pick data
    x = data.train_X;
    y = data.train_Y;
    xt = data.test_X;
    yt = data.test_Y;

    % GP covariance function
    gpcf = gpcf_matern52('lengthScale', 1, 'magnSigma2', 1, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
        
    % Likelihood
    lik = lik_t('nu', 3, 'nu_prior', [], ...
            'sigma2', 1^2, 'sigma2_prior', pn);

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

    % *** EP ***    
    
    % Create the GP sstructure
    gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4, ...
            'latent_method', 'EP');
    
    % Set the options for the optimization
    opt = optimset('TolFun',1e-3,'TolX',1e-3,'display','iter');
    
    % Optimize with the scaled conjugate gradient method
    gp = gp_optim(gp,x,y,'opt',opt);
    
    % Predictions to test points
    [Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, xt, 'yt', yt);
    
    % Mean LPD
    EP_lpd(fold) = mean(lpyt);

    % *** MCMC ***

    % Note that MCMC for latent values requires often more jitter
    gp = gp_set(gp, 'latent_method', 'MCMC', 'jitterSigma2', 1e-6);

    % Sample using default method, that is, surrogate and elliptical slice
    % samplers these samplers are quite robust with default options
    [gp_rec,g,opt]=gp_mc(gp, x, y, 'nsamples', 22000, 'display', 100);

    % Remove burn-in and thin
    gp_rec=thin(gp_rec,2001,2);

    % Make predictions
    [Ef_mc, Varf_mc, lpy_mc, Ey_mc, Vary_mc] = ...
       gp_pred(gp_rec, x, y, xt, 'yt', yt);

    % Mean LPD
    MCMC_lpd(fold) = mean(lpy_mc);
    
  end
  
  
%% Report and save

  % Table
  fprintf('%10s  :      --LA--            --EP--            --MCMC--\n','')
  fprintf('%10s  : %.3f \\pm %.3f  %.3f \\pm %.3f  %.3f \\pm %.3f \n\n',dataset, ...
      mean(LA_lpd),std(LA_lpd), ...
      mean(EP_lpd),std(EP_lpd), ...
      mean(MCMC_lpd),std(MCMC_lpd))
  
  % Save results
  mkdir('result')
  save(sprintf('result/mcmc-%s.mat',dataset),'LA_lpd','EP_lpd', 'MCMC_lpd')

  
