% driver_regime_analysis.m

K = 8;

%exp_tag = 'exp_20_stoch_1256787w8wv1.5';

eval_regimes='true';
smooth_me = 'false';

if strcmp(eval_regimes,'true')
    
    % running average over 0.4 MTU? = 80 time steps dt=0.005
    smooth_me = 'false';
    
    save(['190501_L96ML_regimes_marginalPDFs_',exp_tag,'.mat'],'id_list')
    
    
    if strcmp(smooth_me,'true')
        % smoothing
        dt = round(mean(T_tru(2:end)-T_tru(1:end-1))*1000)/1000;
        smooth_MTU=0.4;
        t_window = round(smooth_MTU/dt);
        
        t_end = length(X_tru);
        for i_file = 1:length(id_list)
            my_id   = char(id_list{i_file});
            eval(['X = X_',my_id,';'])
            X_tally=zeros(K,t_end-t_window+1);
            for t = 1:t_window
                X_tally = X_tally+X(:,t:t+t_end-t_window);
            end
            X_sm = X_tally./t_window;
            eval(['X_',my_id,'_sm = X_sm;'])
            clear X_sm
        end
    else
        for i_file = 1:length(id_list)
            my_id   = char(id_list{i_file});
            eval(['X_',my_id,'_sm = X_',my_id,';'])
        end
    end
    
    
    %% 1. Calculate EOFs and PCs for truth data
    
    % want to calculate EOFs
    Opt_EOF = 0;
    
    [X_tru_EOF,X_tru_PC] = calc_regimes(X_tru_sm',Opt_EOF);
    
    %% 2. Calculate EOFs and PCs for forecasts
    
    for i_file = 1:length(id_list)
        my_id   = char(id_list{i_file});
        eval(['X_fc = X_',my_id,'_sm;'])
        
        [X_fc_EOF,X_fc_PC] = calc_regimes(X_fc',X_tru_EOF);
        
        eval(['X_',my_id,'_EOF = X_fc_EOF;'])
        eval(['X_',my_id,'_PC  = X_fc_PC;'])
    end
    
    %% plot marginal distributions
    
    dx = 0.1;
    pts = 0:dx:35;
    mxp=0;
    for i_file = 1:length(id_list)
        
        my_id   = char(id_list{i_file});
        eval(['X_PC = X_',my_id,'_PC;'])
        
        X_PC_12 = (X_PC(:,1).^2+X_PC(:,2).^2).^0.5;
        [f,xi] = ksdensity(X_PC_12,pts);
        eval(['pdf_PC12_',my_id,' = f;'])
        eval(['x_axis_PC12_',my_id,' = xi;'])
        mxp = max([mxp,max(f)]);
        
        X_PC_34 = (X_PC(:,3).^2+X_PC(:,4).^2).^0.5;
        [f,xi] = ksdensity(X_PC_34,pts);
        eval(['pdf_PC34_',my_id,' = f;'])
        eval(['x_axis_PC34_',my_id,' = xi;'])
        mxp = max([mxp,max(f)]);
        
        save(['190501_L96ML_regimes_marginalPDFs_',exp_tag,'.mat'],...
            ['pdf_PC12_',my_id],['x_axis_PC12_',my_id],['pdf_PC34_',my_id],['x_axis_PC34_',my_id],'-append')
        
    end
else
    load(['190501_L96ML_regimes_marginalPDFs_',exp_tag,'.mat'])
    
end

% plot
figure
set(gcf,'Position',[1080 30 900 370])
              
% % % choose carefully. Group similar schemes
colorvec  = [       0.0      0.0        0.0; %  1. TRU: black
                 %----- vvv dont use ----- vvvvvv
                 0.6422    0.2392    0.7228;  %  2. det: purple < dont use
                 %----- ^^^ dont use ----- ^^^^^^
                 0.2930         0    0.5078;  %  3. S100 series:violet
                 0.4766    0.2487    0.6198; 
                 0.6602    0.4974    0.7318;
                 0.8438    0.7461    0.8438;
                 0.7773    0.0820    0.5195;  %  7. S200 series:magenta
                 0.8555    0.4375    0.5742;
                 %----- vvv dont use ----- vvvvvv
                 0.4250    0.1625    0.0490;  %  9. S500 series:reds
                 0.6375    0.2438    0.0735; 
                 0.8500    0.3250    0.0980; 
                 1         0.3930    0.1186;
                 0.7432    0.5552    0.100;   % 13. S600 series:yellows
                 1         0.83280   0.150; 
                 %----- ^^^ dont use ------ ^^^^^^
                 0.4250    0.1625    0.0490;  % 15. S700 series:reds
                 0.6375    0.2438    0.0735; 
                 0.8500    0.3250    0.0980; 
                 1         0.3930    0.1186;
                 1.0000    0.5469         0;  % 19. S800 series:yellows
                 1.0000    0.6934         0;
                 1.0000    0.8398         0; 
                      0    0.3052    0.5057;  % 22. S700_white series:blues
                      0    0.4470    0.7410; 
                      0    0.5364    0.8892;
                      0    0.7558    1.;
                 0.2796    0.4044    0.1128;  % 26. S800_white series:greens
                 0.5126    0.7414    0.2068;
                 0.7922    0.9458    0.3196;
                 0.45       0.45       0.45];    % 29. poly: grey
% 
id_list_legend_new = id_list_legend;
id_list_legend_new{3} = 'XU-lrg-w*';
id_list_legend_new{4} = 'XU-med-w*';
id_list_legend_new{5} = 'XU-sml-w*';
id_list_legend_new{6} = 'XU-tny-w*';
id_list_legend_new{7} = 'X-sml-w*';
id_list_legend_new{8} = 'X-tny-w*';
id_list_legend_new{22} = 'XU-lrg-w';
id_list_legend_new{23} = 'XU-med-w';
id_list_legend_new{24} = 'XU-sml-w';
id_list_legend_new{25} = 'XU-tny-w';
id_list_legend_new{26} = 'X-med-w';
id_list_legend_new{27} = 'X-sml-w';
id_list_legend_new{28} = 'X-tny-w';
id_list_legend_new{15} = 'XU-lrg-r';
id_list_legend_new{16} = 'XU-med-r';
id_list_legend_new{17} = 'XU-sml-r';
id_list_legend_new{18} = 'XU-tny-r';
id_list_legend_new{19} = 'X-med-r';
id_list_legend_new{20} = 'X-sml-r';
id_list_legend_new{21} = 'X-tny-r';
% 
% %% NOTE new naming convection.
% % if want this plot, need to check selection of indices
% % | Z_2 | = |PC1,PC2|
% % | Z_1 | = |PC3,PC4|
% % so invert order of panels to make more sense
% chosen_id = [1,24,27,17,20,5,8,29];
% for idx_file = 1:length(chosen_id)
%     i_file = chosen_id(idx_file);
%     my_id   = char(id_list{i_file});
%     
%     my_id   = char(id_list{i_file});
%     eval(['f  = pdf_PC12_',my_id,';'])
%     eval(['xi = x_axis_PC12_',my_id,';'])
%     subplot(1,2,2)
%     hold on
%     plot(xi,f,'color',colorvec(i_file,:),'linewidth',2)
% %     xlabel('| (PC1,PC2) |')
%     xlabel('| Z_2 |')
%     ylabel('pdf')
%     set(gca,'Fontsize',12)
%     box on
%     xlim([0 18])
%     ylim([0 0.16])
%     if idx_file==1, text(0,0.168,'(b)','fontsize',14); end
%     
%     my_id   = char(id_list{i_file});
%     eval(['f  = pdf_PC34_',my_id,';'])
%     eval(['xi = x_axis_PC34_',my_id,';'])
%     subplot(1,2,1)
%     hold on
%     plot(xi,f,'color',colorvec(i_file,:),'linewidth',2)
% %     xlabel('| (PC3,PC4) |')
%     xlabel('| Z_1 |')
%     ylabel('pdf')
%     set(gca,'Fontsize',12)
%     box on
%     xlim([0 15])
%     ylim([0 0.14])
%     if idx_file==1, text(0,0.147,'(a)','fontsize',14); end
%     
% end
% 
% subplot(1,2,2)
% legend(id_list_legend_new(chosen_id),'location','northwest')
% 
% exp_tag_new = 'exp_20_stoch_127w8w78v1_5';
% 
% set(gcf,'PaperPositionMode','auto')
% if strcmp(smooth_me,'true')
%     print(gcf,'-djpeg' ,'-r400',['climate_regime_pdfs_sm_',exp_tag_new,'.jpg'])
%     print(gcf,'-depsc2' ,'-r400',['climate_regime_pdfs_sm_',exp_tag_new,'.eps'])
% else
%     print(gcf,'-djpeg' ,'-r400',['climate_regime_pdfs_',exp_tag_new,'_reduced.jpg'])
%     print(gcf,'-depsc2' ,'-r400',['climate_regime_pdfs_',exp_tag_new,'_reduced.eps'])
% end
% 
%     
% 
% % ================= ================= ================= =================
% % calculate some metrics for difference

for i_file = 2:length(id_list)        
    my_id   = char(id_list{i_file});
    eval(['f  = pdf_PC12_',my_id,';'])
    eval(['xi = x_axis_PC12_',my_id,';'])
    HL_vec(i_file,1) = hellinger(xi,f,pdf_PC12_tru);
    KL_vec(i_file,1) = kullbackleibler(xi,f,pdf_PC12_tru);
    
    eval(['f  = pdf_PC34_',my_id,';'])
    eval(['xi = x_axis_PC34_',my_id,';'])
    HL_vec(i_file,2) = hellinger(xi,f,pdf_PC34_tru);
    KL_vec(i_file,2) = kullbackleibler(xi,f,pdf_PC34_tru);
end

figure
% set(gcf,'Position',[1080 30 480 380])
% set(gcf,'Position',[ 504   322   871   424])
set(gcf,'Position',[440    82   651   370])

useme = [22:28,15:21,3:8,29];% 2:length(id_list);%[2:8,15:21];

hold on

i_count=1;
for i_file = useme
    my_id   = char(id_list{i_file});
    
    HL = HL_vec;
    
%     subplot(1,2,1)
    hold on
    scatter(i_count,HL(i_file,2),'^','filled','markeredgecolor',colorvec(i_file,:),'markerfacecolor',colorvec(i_file,:))
    scatter(i_count,HL(i_file,1),'v','filled','markeredgecolor',colorvec(i_file,:),'markerfacecolor',colorvec(i_file,:))
%     title('Hellinger distance')
    set(gca,'xtick',(1:length(useme)))
    set(gca,'xticklabel',(id_list_legend_new(useme)))
    set(gca,'XTickLabelRotation',45)
    xlim([0.5,length(useme)+0.5])
    box on
    set(gca,'fontsize',16)
%     legend('| (PC1,PC2) |','| (PC3,PC4) |','location','northwest')
    
    set(gca,'yscale','log')
    ylim([2.4*10^-4 0.9])
    ylabel('H','fontsize',16)
    
    i_count=i_count+1;
    
end

plot(1:4,  HL(useme(1:4),2)  ,':','color',[0.7 0.7 0.7])
plot(5:7,  HL(useme(5:7),2)  ,':','color',[0.7 0.7 0.7])
plot(8:11, HL(useme(8:11),2) ,':','color',[0.7 0.7 0.7])
plot(12:14,HL(useme(12:14),2),':','color',[0.7 0.7 0.7])
plot(15:18,HL(useme(15:18),2),':','color',[0.7 0.7 0.7])
plot(19:20,HL(useme(19:20),2),':','color',[0.7 0.7 0.7])
plot(1:4,  HL(useme(1:4),1)  ,'--','color',[0.7 0.7 0.7])
plot(5:7,  HL(useme(5:7),1)  ,'--','color',[0.7 0.7 0.7])
plot(8:11, HL(useme(8:11),1) ,'--','color',[0.7 0.7 0.7])
plot(12:14,HL(useme(12:14),1),'--','color',[0.7 0.7 0.7])
plot(15:18,HL(useme(15:18),1),'--','color',[0.7 0.7 0.7])
plot(19:20,HL(useme(19:20),1),'--','color',[0.7 0.7 0.7])

plot([0.5,length(id_list)+0.5],[HL(useme(21),1),HL(useme(21),1)],'--','linewidth',1.5,'color',[0.5 0.5 0.5])
plot([0.5,length(id_list)+0.5],[HL(useme(21),2),HL(useme(21),2)],':','linewidth',1.5,'color',[0.5 0.5 0.5])


legend('| Z_1 |','| Z_2 |','location','northeast')

exp_tag = 'exp_20_stoch_1256787w8wv1_5';

set(gcf,'PaperPositionMode','auto')
if strcmp(smooth_me,'true')
    print(gcf,'-djpeg' ,'-r400',['climate_regime_pdf_metrics_sm_',exp_tag,'.jpg'])
    print(gcf,'-depsc2' ,'-r400',['climate_regime_pdf_metrics_sm_',exp_tag,'.eps'])
else
    print(gcf,'-djpeg' ,'-r400',['climate_regime_pdf_metrics_',exp_tag,'_log.jpg'])
    print(gcf,'-depsc2' ,'-r400',['climate_regime_pdf_metrics_',exp_tag,'_log.eps'])
end

