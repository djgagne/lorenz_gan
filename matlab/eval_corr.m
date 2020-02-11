% % eval_corr.m
%
% no_X=8;

%exp_tag = 'exp_20_stoch_1256787w8wv1.5';

calc_corrs = 'true';

if strcmp(calc_corrs,'true')
    
    chosen_id = 1:length(id_list);
    
    for idx_file = 1:length(chosen_id)
        i_file = chosen_id(idx_file);
        my_id   = char(id_list{i_file});
        
        eval(['my_data_all = X_',my_id,';'])
        
        corrs = corrcoef(my_data_all');
        corrs_reshape = corrs;
        for ind = 2:no_X
            corrs_reshape(:,ind) = [corrs(ind:end,ind);corrs(1:ind-1,ind)];
        end
        corrs_reshape_mn = mean(corrs_reshape,2);
        corrs_reshape_mn = corrs_reshape_mn(1:no_X/2+1);
        eval(['corr_spat_',my_id,' = corrs_reshape;'])
        
        eval(['spat_corrs_',my_id,' = corrs_reshape_mn;'])
        spat_lagvec = 0:(length(corrs_reshape_mn)-1);
        
    end
    
    %% ================= ================= ================= =================
    %%  temporal correlation
    laglist = [1,2,5:10:500];
    
    
    for idx_file = 1:length(chosen_id)
        i_file = chosen_id(idx_file);
        my_id   = char(id_list{i_file});
        
        eval(['my_data_all = X_',my_id,';'])
        eval(['my_time = T_',my_id,';'])
        dt = my_time(2)-my_time(1);
        
        corrs = zeros(length(laglist),no_X);
        for i_lag = 1:length(laglist)
            for ind=1:no_X
                lag = laglist(i_lag);
                R = corrcoef(my_data_all(ind,1:end-lag)',my_data_all(ind,1+lag:end)');
                corrs(i_lag,ind) = R(1,2);
            end
            disp(i_lag)
        end
        
        corrs_mn = mean(corrs,2);
        
        eval(['temp_corrs_',my_id,' = corrs;'])
        temp_lagvec = dt*laglist;
        
    end
    save(['190410_corrs_',exp_tag,'.mat'])
else
    load(['190410_corrs_',exp_tag,'.mat'])
end

%% FIGURE PLOTTING %%

%% choose carefully. Group similar schemes
linecolors = [       0.0      0.0        0.0; %  1. TRU: black
                 %----- vvv dont use ----- vvvvvv
                 0.6422    0.2392    0.7228;  %  2. det: purple < dont use
                 %----- ^^^ dont use ----- ^^^^^^
                 0.3       0.3       0.3;  %  3. S100 series:violet
                 0.3       0.3       0.3 ; 
                 0.3       0.3       0.3 ;
                 0.3       0.3       0.3 ;
                  0.75       0.75       0.75;  %  7. S200 series:magenta
                  0.75       0.75       0.75;
                 %----- vvv dont use ----- vvvvvv
                 0.4250    0.1625    0.0490;  %  9. S500 series:reds
                 0.6375    0.2438    0.0735; 
                 0.8500    0.3250    0.0980; 
                 1         0.3930    0.1186;
                 0.7432    0.5552    0.100;   % 13. S600 series:yellows
                 1         0.83280   0.150; 
                 %----- ^^^ dont use ------ ^^^^^^
                 0.8500    0.2250    0.0980;  % 15. S700 series:reds
                 0.8500    0.2250    0.0980; 
                 0.8500    0.2250    0.0980; 
                 0.8500    0.2250    0.0980; 
                 1.0000    0.5469         0;  % 19. S800 series:yellows
                 1.0000    0.5469         0;
                 1.0000    0.5469         0; 
                       0   0.5364    0.8892;  % 22. S700_white series:blues
                       0   0.5364    0.8892; 
                       0   0.5364    0.8892;
                       0   0.5364    0.8892;
                 0.4    0.7558    1.;  % 26. S800_white series:greens
                 0.4    0.7558    1.;
                 0.4    0.7558    1.;
                 0.5       0.5       0.5];    % 29. poly: grey



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


figure
subplot(1,3,1)
set(gcf,'Position',[1080 30 1100 700])
hold on

chosen_id = [1,24,27,17,20,5,7,29];
for idx_file = 1:length(chosen_id)
    i_file = chosen_id(idx_file);
    my_id   = char(id_list{i_file});
    
    eval(['corrs_reshape_mn = spat_corrs_',my_id,';'])
    
    if i_file==1
        plot(0:(length(corrs_reshape_mn)-1),corrs_reshape_mn,'-o',...
            'linewidth',2,'color',linecolors(i_file,:))
    else
        plot(0:(length(corrs_reshape_mn)-1),corrs_reshape_mn,'-x',...
            'linewidth',1.5,'color',linecolors(i_file,:))
    end
    
end
plot([0 no_X/2],[0 0],'k')
xlabel('spatial lag')
ylabel('correlation')
set(gca,'Fontsize',14)
text(0, 1.075,'(a)','fontsize',14)
box on
title('Spatial correlation')

%% ================= ================= ================= =================
%%  temporal correlation
laglist = [1,2,5:10:500];

% figure
set(gcf,'Position',[1080 30 1280 400])
subplot(1,3,2:3)
hold on

for idx_file = 1:length(chosen_id)
    i_file = chosen_id(idx_file);
    my_id   = char(id_list{i_file});
        
    eval(['corrs_mn = mean(temp_corrs_',my_id,',2);'])
    
    if i_file==1
        plot(dt*laglist,corrs_mn,'-o','linewidth',2,'color',linecolors(i_file,:))
    else
        plot(dt*laglist,corrs_mn,'-x','linewidth',2,'color',linecolors(i_file,:))
    end
    
end
plot([0 dt*laglist(end)],[0 0],'k')
xlim([0 dt*laglist(end)])
xlabel('temporal lag / MTU')
% ylabel('correlation')
text(0, 1.075,'(b)','fontsize',14)
set(gca,'Fontsize',14)
box on
title('Temporal correlation')
legend(id_list_legend_new(chosen_id),'fontsize',12)

exp_tag_fig = 'exp_20_stoch_127w8w78v1_5';

set(gcf,'PaperPositionMode','auto')
print(gcf,'-djpeg'  ,'-r400',['climate_spat_temp_corr_',exp_tag_fig,'.jpg'])
print(gcf,'-depsc2' ,'-r400',['climate_spat_temp_corr_',exp_tag_fig,'.eps'])


