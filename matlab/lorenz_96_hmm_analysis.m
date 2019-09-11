% HMM analysis uses the HMM 'HMMall' toolbox of Kevin Murphy downloaded
% from
% https://www.cs.ubc.ca/~murphyk/Software/HMM/hmm_download.html

tic
rand('state',sum(100*clock));
randn('state',sum(100*clock));
addpath ~/work/matlab
addpath(genpath('/Volumes/external/data/machine_learning/hmm/HMMall'))




flnm{1} = '/Users/monahana/work/data/lorenz_output.nc';

flnm{2} = 'gan_700_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{3} = 'gan_701_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{4} = 'gan_702_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{5} = 'gan_703_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{6} = 'gan_801_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{7} = 'gan_802_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{8} = 'gan_803_climate_white/02000000/lorenz_forecast_02000000_00.nc';
flnm{9} = 'gan_700_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{10} = 'gan_701_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{11} = 'gan_702_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{12} = 'gan_703_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{13} = 'gan_801_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{14} = 'gan_802_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{15} = 'gan_803_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{16} = 'gan_100_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{17} = 'gan_101_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{18} = 'gan_102_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{19} = 'gan_103_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{20} = 'gan_202_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{21} = 'gan_203_climate/02000000/lorenz_forecast_02000000_00.nc';
flnm{22} = 'poly_add_climate/lorenz_forecast_02000000_00.nc';

xnm{1} = 'lorenz_x';
xnm{2} = 'x';
xnm{3} = 'x';
xnm{4} = 'x';
xnm{5} = 'x';
xnm{6} = 'x';
xnm{7} = 'x';
xnm{8} = 'x';
xnm{9} = 'x';
xnm{10} = 'x';
xnm{11} = 'x';
xnm{12} = 'x';
xnm{13} = 'x';
xnm{14} = 'x';
xnm{15} = 'x';
xnm{16} = 'x';
xnm{17} = 'x';
xnm{18} = 'x';
xnm{19} = 'x';
xnm{20} = 'x';
xnm{21} = 'x';
xnm{22} = 'x';



for input_data = 1:22
    for kk=1:1
        
        % HMM optimization not deterministic, setting kk > 1 generates
        % ensemble of HMM results for each input dataset
        kk
        
        ncid = netcdf.open(flnm{input_data});
        vrnm = netcdf.inqVarID(ncid,xnm{input_data});
        x = netcdf.getVar(ncid,vrnm);
        netcdf.close(ncid);
        
        
        nclust=2;
        nvr = 8;
        
        
        npt = length(x(1,:));
        
        x_anom = x-mean(x,2)*ones(1,npt);
        
        clear u
        clear v
        for j=1:4
            s = cos(2*pi*j*(1:8)/8);
            s = s/sqrt(sum(s.^2));
            c = sin(2*pi*j*(1:8)/8);
            c = c/sqrt(sum(c.^2));
            u(j,:) = s*x_anom;
            v(j,:) = c*x_anom;
        end
        
        clear x
        clear s
        clear c
        for j=1:4
            x(j,:) = sqrt(u(j,:).^2+v(j,:).^2);
        end
        
        nvr = 4;
        
        options = statset('Display','final','maxiter',500);
        clear tmp
        obj = gmdistribution.fit(x',nclust,'Options',options);
        tmp = obj.mu;
        
        
        for jj=1:nclust
            
            m{jj} = tmp(jj,:);
            
        end
        
        tmp = obj.Sigma;
        
        for jj=1:nclust
            
            s{jj} = tmp(:,:,jj);
            
        end
        
        clear mu0;
        clear sigma0;
        
        for jj=1:nclust
            mu0(:,jj) = m{jj};
            
            sigma0(:,:,jj) = reshape(s{jj},[nvr nvr 1]);
        end
        
        prior0 = normalise(rand(nclust,1));
        transmat0 = mk_stochastic(rand(nclust,nclust));
        
        [ll,prior1,transmat1,mu1,sigma1] = ...
            mhmm_em(x,prior0,transmat0,mu0,sigma0,[],'max_iter',50);
        
        pth{input_data,kk} = zeros(1,npt);
        
        B = mixgauss_prob(x,mu1,sigma1);
        pth{input_data,kk} = viterbi_path(prior1,transmat1,B);
        
        [f{input_data,kk},r1{input_data,kk},r2{input_data,kk}] = gaussian_2d_kernel_estimator(x(1:2,:),.5,[0 20],[0 20],6);
        
        pts1 = find(pth{input_data,kk}==1);
        pts2 = find(pth{input_data,kk}==2);
        
        [g1{input_data,kk},r1{input_data,kk},r2{input_data,kk}] = gaussian_2d_kernel_estimator(x(1:2,pts1),.5,[0 20],[0 20],6);
        [g2{input_data,kk},r1{input_data,kk},r2{input_data,kk}] = gaussian_2d_kernel_estimator(x(1:2,pts2),.5,[0 20],[0 20],6);
        g1{input_data,kk} = g1{input_data,kk}*length(pts1)/length(pth{input_data,kk});
        g2{input_data,kk} = g2{input_data,kk}*length(pts2)/length(pth{input_data,kk});
        
        Q{input_data,kk} = transmat1;
        mu_out{input_data,kk} = mu1;
        sig_out{input_data,kk} = sigma1;
    end
end

save lorenz_96_hmm_out.mat

toc