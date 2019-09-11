%  % eval_pdf.m
%  

%% final for paper

%exp_tag  = 'exp_20_stoch_1256787w8wv1.5';

 
 %% let's start with a PDF!
 calc_pdf = 'true';
 
 if strcmp(calc_pdf,'true')
     disp('calc PDF')
      
     no_X = size(X_tru,1);
     dx = 0.1;
     pts = floor(mn):dx:ceil(mx);
     mxp=0;
     for i_file = 1:length(id_list)
         
         my_id   = char(id_list{i_file});
         
         eval(['my_data_all = X_',my_id,';'])
         for ind=1:no_X
             my_data = my_data_all(ind,:);
             [f,xi] = ksdensity(my_data,pts);
             eval(['pdf_',num2str(ind),'_',my_id,' = f;'])
             eval(['x_axis_',num2str(ind),'_',my_id,' = xi;'])
             mxp = max([mxp,max(f)]);
             disp(ind)
         end
         
         my_data = reshape(my_data_all,no_X*size(my_data_all,2),1);
         [f,xi] = ksdensity(my_data,pts);
         eval(['pdf_',my_id,' = f;'])
         eval(['x_axis_',my_id,' = xi;'])
         mxp = max([mxp,max(f)]);
         
         disp(['  ',num2str(i_file)])
         
     end
     
     save(['180706_pdf_data_',exp_tag,'.mat'])
 else
     load(['180706_pdf_data_',exp_tag,'.mat'])
 end
 %     



%% ================= ================= ================= =================
%% and plot it
% 
% % % choose carefully. Group similar schemes
linecolors = [       0.0      0.0        0.0; %  1. TRU: black
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
                 0.05       0.05       0.05];    % 29. poly: grey


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
mxp = 0.06;
set(gcf,'Position',[200 400 990 380])
chosen_id = [1,24,27,17,20,5,7,29];
hold on
for idx_file = 1:length(chosen_id)
   i_file = chosen_id(idx_file);
   my_id   = char(id_list{i_file});
%     for ind=1:no_X
%         eval(['f  = pdf_',num2str(ind),'_',my_id,';'])
%         eval(['xi = x_axis_',num2str(ind),'_',my_id,';'])
%         subplot(5,2,ind+2)
%         hold on
%         plot(xi,f,'linewidth',2,'color',linecolors(i_file,:))
%     end
%     
   eval(['f  = pdf_',my_id,';'])
   eval(['xi = x_axis_',my_id,';'])
   
   subplot(1,2,1)
   hold on
   if i_file == 1
       % plot black shaded region to indicate variability between X
       plot(xi,f,'--','linewidth',2,'color',linecolors(i_file,:))
       eval(['curve1 = min([pdf_1_',my_id,';pdf_2_',my_id,';pdf_3_',my_id,';pdf_4_',my_id,';',...
                           'pdf_5_',my_id,';pdf_6_',my_id,';pdf_7_',my_id,';pdf_8_',my_id,'],[],1)'])
       eval(['curve2 = max([pdf_1_',my_id,';pdf_2_',my_id,';pdf_3_',my_id,';pdf_4_',my_id,';',...
                           'pdf_5_',my_id,';pdf_6_',my_id,';pdf_7_',my_id,';pdf_8_',my_id,'],[],1)'])
%         plot(xi, curve1, 'color',[0.8 0.8 0.8], 'LineWidth', 1);
%         plot(xi, curve2, 'color',[0.8 0.8 0.8], 'LineWidth', 1);
       xi2 = [xi, fliplr(xi)];
       inBetween = [curve1, fliplr(curve2)];
       fill(xi2, inBetween,[0.8 0.8 0.8] ,'edgecolor',[0.8 0.8 0.8],'facealpha',0.5,'edgealpha',0.5);
       
   else
       plot(xi,f,'linewidth',2,'color',linecolors(i_file,:))
   end
   box on
   xlabel('X')
   ylabel('p(X)')
   set(gca,'fontsize',16)
   xlim([-15 20])
   ylim([0 0.085])
   text(-15,0.0893,'(a)','fontsize',16)
   
   subplot(1,2,2)
   hold on
   if i_file == 1
       plot(xi,f-pdf_tru,'--','linewidth',2,'color',linecolors(i_file,:))
       % plot black shaded region to indicate variability between X
       eval(['curve1 = min([f-pdf_1_',my_id,';f-pdf_2_',my_id,';f-pdf_3_',my_id,';f-pdf_4_',my_id,';',...
                           'f-pdf_5_',my_id,';f-pdf_6_',my_id,';f-pdf_7_',my_id,';f-pdf_8_',my_id,'],[],1)'])
       eval(['curve2 = max([f-pdf_1_',my_id,';f-pdf_2_',my_id,';f-pdf_3_',my_id,';f-pdf_4_',my_id,';',...
                           'f-pdf_5_',my_id,';f-pdf_6_',my_id,';f-pdf_7_',my_id,';f-pdf_8_',my_id,'],[],1)'])
%         plot(xi, curve1, 'color', [0.8 0.8 0.8], 'LineWidth', 1);
%         plot(xi, curve2, 'color', [0.8 0.8 0.8], 'LineWidth', 1);
       xi2 = [xi, fliplr(xi)];
       inBetween = [curve1, fliplr(curve2)];
       fill(xi2, inBetween,[0.8 0.8 0.8] ,'edgecolor',[0.8 0.8 0.8],'facealpha',0.5,'edgealpha',0.5);
       
   else
       plot(xi,f-pdf_tru,'linewidth',2,'color',linecolors(i_file,:))
   end
   box on
   ylabel('p(X)_{model} - p(X)_{truth}')
   xlabel('X')
   set(gca,'fontsize',16)
   xlim([-15 20])
   ylim([-0.02 0.02])
   text(-15,0.022,'(b)','fontsize',16)
   
end
id_list_legend_tmp = cell(1,length(id_list_legend_new(chosen_id))+1);
id_list_legend_tmp(1) = id_list_legend_new(1);
id_list_legend_tmp(2) = {'tru var'};
id_list_legend_tmp(3:end) = id_list_legend_new(chosen_id(2:end));

legend(id_list_legend_tmp,'fontsize',14,'location','northeastoutside')


exp_tag_fig = 'exp_20_stoch_127w8w78v1.5';

set(gcf,'PaperPositionMode','auto')
print(gcf,'-djpeg' ,'-r400',['climate_pdf_simple2_',exp_tag_fig,'.jpg'])
print(gcf,'-depsc2' ,'-r400',['climate_pdf_simple2_',exp_tag_fig,'.eps'])

%% ================= ================= ================= =================
%% calculate some metrics for difference
colorvec = linecolors;

for i_file = 1:length(id_list)        
   my_id   = char(id_list{i_file});
   eval(['f  = pdf_',my_id,';'])
   eval(['xi = x_axis_',my_id,';'])
   HL_vec(1,1) = hellinger(xi,f,pdf_tru);
   KL_vec(1,1) = kullbackleibler(xi,f,pdf_tru);
   for ind=1:no_X
       eval(['f  = pdf_',num2str(ind),'_',my_id,';'])
       eval(['xi = x_axis_',num2str(ind),'_',my_id,';'])
       eval(['HL_vec(1+ind,1) = hellinger(xi,f,pdf_',num2str(ind),'_tru);'])
       eval(['KL_vec(1+ind,1) = kullbackleibler(xi,f,pdf_',num2str(ind),'_tru);'])
   end
   
   eval(['HL_',my_id,' = HL_vec;'])
   eval(['KL_',my_id,' = KL_vec;'])
end

figure
set(gcf,'Position',[ 504   322   871   424])

%% plot just a few
hold on
plot(1:4,[HL_Sgan700w(1),HL_Sgan701w(1),HL_Sgan702w(1),HL_Sgan703w(1)],'color',[0.8 0.8 0.8])
plot(5:7,[HL_Sgan801w(1),HL_Sgan802w(1),HL_Sgan803w(1)],'color',[0.8 0.8 0.8])
plot(8:11,[HL_Sgan700(1),HL_Sgan701(1),HL_Sgan702(1),HL_Sgan703(1)],'color',[0.8 0.8 0.8])
plot(12:14,[HL_Sgan801(1),HL_Sgan802(1),HL_Sgan803(1)],'color',[0.8 0.8 0.8])
plot(15:18,[HL_Sgan100(1),HL_Sgan101(1),HL_Sgan102(1),HL_Sgan103(1)],'color',[0.8 0.8 0.8])
plot(19:20,[HL_Sgan202(1),HL_Sgan203(1)],'color',[0.8 0.8 0.8])

useme = [22:28,15:21,3:8,29];% 2:length(id_list);%[2:8,15:21];
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

% id_list_legend_new = id_list_legend;
i_count=1;
hold on

plot([0,length(id_list)],[HL_Spolyadd(1),HL_Spolyadd(1)],'--','linewidth',1.5,'color',[0.5 0.5 0.5])
  

for i_file = useme
   my_id   = char(id_list{i_file});
   
   eval(['HL = HL_',my_id,';'])
   eval(['KL = KL_',my_id,';'])
   
%    subplot(1,2,1)
   hold on
   scatter(i_count.*ones(no_X,1),HL(2:end),'x','markeredgecolor',colorvec(i_file,:))
   scatter(i_count,HL(1),'o','filled','markeredgecolor',colorvec(i_file,:),'markerfacecolor',colorvec(i_file,:))
   title('Hellinger distance')
   set(gca,'Xtick',(1:length(id_list)))
   set(gca,'XTickLabels',(id_list_legend_new(useme)))
   set(gca,'XTickLabelRotation',45)
   xlim([0.5,length(useme)+0.5])
   box on
   set(gca,'fontsize',14)
   set(gca,'YScale','log')
%    
%    subplot(1,2,2)
%    hold on
%    scatter(i_count.*ones(no_X,1),KL(2:end),'x','markeredgecolor',colorvec(i_file,:))
%    scatter(i_count,KL(1),'o','filled','markeredgecolor',colorvec(i_file,:),'markerfacecolor',colorvec(i_file,:))
%    title('Kullback Leibler Divergence')
%    set(gca,'Xtick',(1:length(useme)))
%    set(gca,'XTickLabels',(id_list_legend_new(useme)))
%    set(gca,'XTickLabelRotation',45)
%    xlim([0.5,length(useme)+0.5])
%    box on
%    set(gcf,'Position',[1080 30 1030 380])
%         
%    set(gca,'fontsize',14)
%    set(gca,'YScale','log')
   i_count=i_count+1;
   
end

ylim([2*10^(-4) 2*10^(-1)])

% exp_tag_new = exp_tag;
exp_tag_fig = 'exp_20_stoch_127w8w78v1.5';

set(gcf,'PaperPositionMode','auto')
print(gcf,'-djpeg' ,'-r400',['climate_pdf_metrics_simple_',exp_tag_fig,'.jpg'])
print(gcf,'-depsc2' ,'-r400',['climate_pdf_metrics_simple_',exp_tag_fig,'.eps'])

