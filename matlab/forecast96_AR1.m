 % forecast96_AR1

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % STRATONOVICH VERSION                                           %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% script to solve AN APPROXIMATION to the equations:

% dX_k / dt = -X_(k-1) ( X_(k-2) - X_(k+1) ) -X_k -(hc/b)*sum_j (Y_j,k) + F
% dY_j,k / dt = -cbY_(j+1,k) ( X_(j+2,k) - Y_(j-1,k) ) -cY_j,k + (hc/b)X_k

% where the Y variables are parametrized such that:

% dX*_k / dt = -X*_(k-1) ( X*_(k-2) - X*_(k+1) ) -X*_k + F - g_U(X*)
%where g_U is a parametrization of the effects of Y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EITHER  a) run for many init cond & ens, for 3 units/15 days each %
% OR      b) run single realisation out for 2-3 yrs => 75 units     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GLOBAL VARIABLES:

global paramf ef_past time s1


% parameters:

% DEFAULT VALUES:
% PERSISTENT STOCHASTIC
phi_meas =  phi_meas_AR1; 

sd_meas  = sd_meas_A; 

phi = phi_meas;
sd  = sd_meas;

dtf = 0.005;           % timestep
t_tot = 160;
mf  = t_tot/dtf;       % no. iterations - so covers same time as lorenz96
n_ens = 40;            % no. members in ensemble forecast - CASE (a)
%n_ens = 1;             % no. members in ensemble forecast - CASE (b)
no_samp = 300;          % number of samples - CASE (a)
%no_samp = 1;           % no. samples      - CASE (b)
t_fcast = 1.2;           % (a) forecast 3 time units = 15 days
%t_fcast = 600;          % (b) forecast 200 time units = 1000d, 2.7yrs

param_f = [phi, sd, dtf, mf]; %to save to file


%% random number generation
%[s1,s2,s3]=RandStream.create('mlfg6331_64','NumStreams',3);

s1=RandStream('mt19937ar','seed',1);

%% Vf matrix Vf(t,k) - each col. is the time series of Xf_k


%ASSIGN FUNCTION HANDLE
Vfdot_han = @Vfdot_AR1;
options = odeset('InitialStep',dtf,'MaxStep',dtf);

%tspan = [0, t_tot];


Phi = [0 0.368 0.607 0.779 0.882 0.939 0.969 phi_meas 0.993 0.996 0.998];
%Phi = [0.996 0.998];
%Phi = phi_meas;

SD = sd_meas.*[0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5];
%SD = sd_meas.*[2.75 3];
%SD = sd_meas;


for sd = SD
%     
%         if sd == sd_meas*2.75
%             Phi = [0.996 0.998];
%         else
%             Phi = [0 0.368 0.607 0.779 0.882 0.939 0.969 phi_meas 0.992 0.996 0.998];
%         end
        
    
    for phi = Phi
        
     
        % initialise matrices
        ef_past = zeros(K,1);
        Xf = zeros((t_fcast/dtf)+1,K,no_samp,n_ens);  % (a) (time, X var, init cond, n_i)
       
        %determ.:
        %paramf = [K, F, bd_0, bd_1, bd_2, bd_3, bd_4, bd_5, phi, sd, dtf, mf]; % global vars
        % stoch AR1: keep as cubic for time being.
        paramf = [K, F, bs2_0, bs2_1, bs2_2, bs2_3, phi, sd, dtf, mf]; % global vars
     
        
        for n_i = 1:no_samp
            % initial conditions:
            Xf_0 = V_init(n_i,1:8);            % initialise
            Vf_0 = Xf_0;
            Vf = zeros(t_fcast/dtf+1, K);
            Vf(1,:) = Xf_0;
            Tf   = zeros(t_fcast/dtf+1, 1);
            tfspan = 0:dtf:t_fcast;
            n_i
            
            
            % to save doing ensemble for deterministic case :)
            if sd == 0
                n_ens_sd = 1;
            else
                n_ens_sd = n_ens;
            end

           
            for ens = 1:n_ens_sd
                
                % keeps track of the time in ode23
                time = -dtf;
                ef_past  = sd.*randn(s1,K,1);
                [Tf,Vf_AR1] = ode23(Vfdot_han, tfspan, Vf_0, options);
                Xf(:,:,n_i,ens) = Vf_AR1(:,:); % CASE (a)
                %Xf(:,:) = Vf(:,:);              % CASE (b)
                
                %ens
            end
            
        end
        
        Xf_params = [sd, phi];
        
        % such that determinsitic case takes up same amount of room as
        % others
        if sd == 0
            Xf_tmp = Xf;
            Xf = zeros((t_fcast/dtf)+1,K,no_samp,n_ens);
            for ens = 1:n_ens
                Xf(:,:,:,ens) = Xf_tmp(:,:,:,1);
            end
            Xf_tmp = [];
        end
        
% CASE (a):

        % coarsen data before write to file ***ONLY IN WEATHER MODE***
        if t_fcast < 5
            csn_i = 1;
            Xf_write = zeros(t_fcast/dtf/5+1,K,no_samp,n_ens);
            for t=1:(t_fcast/dtf)+1
                if mod(t-1,5) == 0
                    Xf_write(csn_i,:,:,:) = Xf(t,:,:,:);
                    csn_i = csn_i+1;
                end
            end
        else
            Xf_write = Xf;
        end
        
        
        
        % CASE (a):
        
        fid1 = fopen('fcast_Xf_AR1_init300_dtmax.bin','a');
        fwrite(fid1, Xf_write, 'double');
        fclose(fid1);
        
        fid2 = fopen('fcast_params_AR1_init300_dtmax.bin','a');
        fwrite(fid2, Xf_params, 'double');
        fclose(fid2);
        
        % CASE (b):
        
%         fid1 = fopen('fcast_Xf_AR1_clim_new.bin','a');
%         fwrite(fid1, Xf_write, 'double');
%         fclose(fid1);
%         
%         fid2 = fopen('fcast_params_AR1_clim_new.bin','a');
%         fwrite(fid2, Xf_params, 'double');
%         fclose(fid2);        
%          
%         Xf = []; %clear array
        
        phi
        
    end
    
    sd

end
  



%set up truth vector, X_tru. coarsen to match forecast resolution

%% X TRU COARSE HERE!!!!
% 
n_i = 1;
i   = 0;
while n_i <= no_samp
    if mod(i,t_samp) == 0
        for t=0:t_fcast/dtf/5
            X_tru_coarse(t+1,:,n_i) = X((i+1+t*(5*dtf/dt)),:);
            T_tru_coarse(t+1,1) = t*dtf*5;
        end
        n_i = n_i + 1;
    end
    i=i+1;
end





%settings:
% 
% fid1 = fopen('initialise.bin','a');         % add data to end of file
% fwrite(fid1, param_f, 'double');
% fclose(fid1);

%results:

% fid2 = fopen('forecast.bin','w');
% fwrite(fid2, Vf, 'double');
% fwrite(fid2, Tf, 'double');
% fclose(fid2);

% fid1 = fopen('fcast_sdphi3.bin','w');
% fwrite(fid1, Xf_params, 'double');
% fwrite(fid1, Xf, 'double');
% fclose(fid1);




