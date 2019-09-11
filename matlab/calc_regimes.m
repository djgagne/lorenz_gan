function [Xf_Eig_Vecs,Xf_PCs] = calc_regimes(Xf_read,Opt_EOF)

K=size(Xf_read,2);

% 0. Ensure all X variables perfectly interchangeable

Xf_big = [Xf_read(:,1:K);
          Xf_read(:,2:K),Xf_read(:,1);
          Xf_read(:,3:K),Xf_read(:,1:2);
          Xf_read(:,4:K),Xf_read(:,1:3);
          Xf_read(:,5:K),Xf_read(:,1:4);
          Xf_read(:,6:K),Xf_read(:,1:5);
          Xf_read(:,7:K),Xf_read(:,1:6);
          Xf_read(:,K)  ,Xf_read(:,1:7)];

%Xf_big = Xf_read;
      
% 1. calculate the time mean of each column.

Xf_time_mean = mean(Xf_big);

% 2. Remove time mean to create anomoly matrix

Xf_Anom     = Xf_read - ones(length(Xf_read),1)*Xf_time_mean;
Xf_Anom_big = Xf_big  -  ones(length(Xf_big),1)*Xf_time_mean;

%% -- if we have supplied EOFS, calculate PCs in this EOF space
if Opt_EOF==0
    % need to calculate EOFs
    
    % 3. Calculate covariance matrix.
    Xf_Cov = Xf_Anom_big'*Xf_Anom_big;
    
    % 4. Calculate eigenvalues and eigenvectors - smallest to largest.
    [Xf_Eig_Vecs,Xf_Eig_Vals] = eig(Xf_Cov);
    
    % Eigenvector matrix ordered wrong - re-order to give EOFs;
    EOF_ind = K:-1:1;
    for k=1:K
        tmp1(:,EOF_ind(k)) = Xf_Eig_Vecs(:,k);
        tmp2(EOF_ind(k),EOF_ind(k)) = Xf_Eig_Vals(k,k);
    end
    Xf_Eig_Vecs = tmp1;
    Xf_Eig_Vals = tmp2;
    tmp1 =[];
    tmp2 =[];
    
    % 6. Fraction of variance explained by each EOF:
    Tot_eigs = sum(diag(Xf_Eig_Vals));
    Xf_FracVar = zeros(K,1);
    for k=1:K
        Xf_FracVar(k) = Xf_Eig_Vals(k,k)/Tot_eigs;
    end
    
%     figure
%     plot(Xf_FracVar,'-x','linewidth',2)
%     xlabel('EOF')
%     ylabel('Variance Explained')
%     box on
%     set(gca,'fontsize',12)
%     set(gcf,'PaperPositionMode','auto')
%     print(gcf,'-djpeg'  ,'-r400','climate_regimes_fracvar.jpg')
%     print(gcf,'-depsc2' ,'-r400','climate_regimes_fracvar.eps')
    
else
    Xf_Eig_Vecs = Opt_EOF;
end


% 7. Principle Components
Xf_PCs = Xf_Anom*Xf_Eig_Vecs;


