clear all;
close all;

% 

load rmse_rpss_gan_100_exp_20_stoch.mat
rmse_gans_100 = rmse;
sprd_gans_100 = sprd;
rpss_gans_100 = rpss_mn;
rps_gans_100  = rps_mn;

load rmse_rpss_gan_101_exp_20_stoch.mat
rmse_gans_101 = rmse;
sprd_gans_101 = sprd;
rpss_gans_101 = rpss_mn;
rps_gans_101  = rps_mn;

load rmse_rpss_gan_102_exp_20_stoch.mat
rmse_gans_102 = rmse;
sprd_gans_102 = sprd;
rpss_gans_102 = rpss_mn;
rps_gans_102  = rps_mn;

load rmse_rpss_gan_103_exp_20_stoch.mat
rmse_gans_103 = rmse;
sprd_gans_103 = sprd;
rpss_gans_103 = rpss_mn;
rps_gans_103  = rps_mn;

load rmse_rpss_gan_202_exp_20_stoch.mat
rmse_gans_202 = rmse;
sprd_gans_202 = sprd;
rpss_gans_202 = rpss_mn;
rps_gans_202  = rps_mn;

load rmse_rpss_gan_203_exp_20_stoch.mat
rmse_gans_203 = rmse;
sprd_gans_203 = sprd;
rpss_gans_203 = rpss_mn;
rps_gans_203  = rps_mn;


load rmse_rpss_gan_700_white_exp_20_stoch.mat
rmse_gans_700w = rmse;
sprd_gans_700w = sprd;
rpss_gans_700w = rpss_mn;
rps_gans_700w  = rps_mn;

load rmse_rpss_gan_701_white_exp_20_stoch.mat
rmse_gans_701w = rmse;
sprd_gans_701w = sprd;
rpss_gans_701w = rpss_mn;
rps_gans_701w  = rps_mn;

load rmse_rpss_gan_702_white_exp_20_stoch.mat
rmse_gans_702w = rmse;
sprd_gans_702w = sprd;
rpss_gans_702w = rpss_mn;
rps_gans_702w  = rps_mn;

load rmse_rpss_gan_703_white_exp_20_stoch.mat
rmse_gans_703w = rmse;
sprd_gans_703w = sprd;
rpss_gans_703w = rpss_mn;
rps_gans_703w  = rps_mn;

load rmse_rpss_gan_801_white_exp_20_stoch.mat
rmse_gans_801w = rmse;
sprd_gans_801w = sprd;
rpss_gans_801w = rpss_mn;
rps_gans_801w  = rps_mn;

load rmse_rpss_gan_802_white_exp_20_stoch.mat
rmse_gans_802w = rmse;
sprd_gans_802w = sprd;
rpss_gans_802w = rpss_mn;
rps_gans_802w  = rps_mn;

load rmse_rpss_gan_803_white_exp_20_stoch.mat
rmse_gans_803w = rmse;
sprd_gans_803w = sprd;
rpss_gans_803w = rpss_mn;
rps_gans_803w  = rps_mn;

% 
load rmse_rpss_gan_700_exp_20_stoch.mat
rmse_gans_700 = rmse;
sprd_gans_700 = sprd;
rpss_gans_700 = rpss_mn;
rps_gans_700  = rps_mn;

load rmse_rpss_gan_701_exp_20_stoch.mat
rmse_gans_701 = rmse;
sprd_gans_701 = sprd;
rpss_gans_701 = rpss_mn;
rps_gans_701  = rps_mn;

load rmse_rpss_gan_702_exp_20_stoch.mat
rmse_gans_702 = rmse;
sprd_gans_702 = sprd;
rpss_gans_702 = rpss_mn;
rps_gans_702  = rps_mn;

load rmse_rpss_gan_703_exp_20_stoch.mat
rmse_gans_703 = rmse;
sprd_gans_703 = sprd;
rpss_gans_703 = rpss_mn;
rps_gans_703  = rps_mn;

load rmse_rpss_gan_801_exp_20_stoch.mat
rmse_gans_801 = rmse;
sprd_gans_801 = sprd;
rpss_gans_801 = rpss_mn;
rps_gans_801  = rps_mn;

load rmse_rpss_gan_802_exp_20_stoch.mat
rmse_gans_802 = rmse;
sprd_gans_802 = sprd;
rpss_gans_802 = rpss_mn;
rps_gans_802  = rps_mn;

load rmse_rpss_gan_803_exp_20_stoch.mat
rmse_gans_803 = rmse;
sprd_gans_803 = sprd;
rpss_gans_803 = rpss_mn;
rps_gans_803  = rps_mn;

load ./rmse_rpss_poly_add_exp_20_stoch.mat
rmse_poly_add = rmse;
sprd_poly_add = sprd;
rpss_poly_add = rpss_mn;
rps_poly_add  = rps_mn;



%% Figure

linecolors = [%--- S100 series:purple ---

                 0.2930         0    0.5078; 

                 0.4766    0.2487    0.6198; 

                 0.6602    0.4974    0.7318;

                 0.8438    0.7461    0.8438;


                 %--- S200 series:pink ---

                 0.7773    0.0820    0.5195;

                 0.8555    0.4375    0.5742;


                 %--- S700_white series:blues ---
                      0    0.3052    0.5057;

                      0    0.4470    0.7410; 

                      0    0.5364    0.8892;

                      0    0.7558    1.;


                 %---  %S800_white series:greens

                 0.2796    0.4044    0.1128;

                 0.5126    0.7414    0.2068;

                 0.7922    0.9458    0.3196;


                 %--- S700 series:reds ---

                 0.4250    0.1625    0.0490;

                 0.6375    0.2438    0.0735; 

                 0.8500    0.3250    0.0980; 

                 1         0.3930    0.1186;


                 %--- S800 series:yellows ---
                 1.0000    0.5469         0;

                 1.0000    0.6934         0;

                 1.0000    0.8398         0;


                  %--- poly: grey

                 0.5       0.5       0.5]; 
             

                
cmap = linecolors;


%% Figures

% Make one plot with all the runs (subset)


cmap = linecolors;

lnwd = 4;
mksz = 8;

mp = get(0,'MonitorPositions');
scrsz = squeeze(mp(1,:));
hf = figure(1);
set(hf,'Color','white','position',[1 scrsz(4)/1 scrsz(3)/1 scrsz(4)/1]);

p1=plot(0:0.005:(2-0.005),rmse_gans_102,'-','Color', cmap(3, :),'LineWidth',lnwd);
hold on;
p2=plot(0:10*0.005:(2-0.005),rmse_gans_102(1:10:end),'-*','Color', cmap(3, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p3=plot(0:0.005:(2-0.005),sprd_gans_102,'--','Color', cmap(3, :),'LineWidth',lnwd);
hold on;

p4=plot(0:0.005:(2-0.005),rmse_gans_202,'-','Color', cmap(5, :),'LineWidth',lnwd);
hold on;
p5=plot(0:10*0.005:(2-0.005),rmse_gans_202(1:10:end),'-*','Color', cmap(5, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p6=plot(0:0.005:(2-0.005),sprd_gans_202,'--','Color', cmap(5, :),'LineWidth',lnwd);
hold on;


p7=plot(0:0.005:(2-0.005),rmse_gans_702w,'-','Color', cmap(9, :),'LineWidth',lnwd);
hold on;
p8=plot(0:10*0.005:(2-0.005),rmse_gans_702w(1:10:end),'-*','Color', cmap(9, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p9=plot(0:0.005:(2-0.005),sprd_gans_702w,'--','Color', cmap(9, :),'LineWidth',lnwd);
hold on;

p10=plot(0:0.005:(2-0.005),rmse_gans_802w,'-','Color', cmap(12, :),'LineWidth',lnwd);
hold on;
p11=plot(0:10*0.005:(2-0.005),rmse_gans_802w(1:10:end),'-*','Color', cmap(12, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p12=plot(0:0.005:(2-0.005),sprd_gans_802w,'--','Color', cmap(12, :),'LineWidth',lnwd);
hold on;

p13=plot(0:0.005:(2-0.005),rmse_gans_702,'-','Color', cmap(16, :),'LineWidth',lnwd);
hold on;
p14=plot(0:10*0.005:(2-0.005),rmse_gans_702(1:10:end),'-*','Color', cmap(16, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p15=plot(0:0.005:(2-0.005),sprd_gans_702,'--','Color', cmap(16, :),'LineWidth',lnwd);
hold on;

p16=plot(0:0.005:(2-0.005),rmse_gans_802,'-','Color', cmap(19, :),'LineWidth',lnwd);
hold on;
p17=plot(0:10*0.005:(2-0.005),rmse_gans_802(1:10:end),'-*','Color', cmap(19, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p18=plot(0:0.005:(2-0.005),sprd_gans_802,'--','Color', cmap(19, :),'LineWidth',lnwd);
hold on;

p19=plot(0:0.005:(2-0.005),rmse_poly_add,'-','Color', cmap(21, :),'LineWidth',lnwd);
hold on;
p20=plot(0:10*0.005:(2-0.005),rmse_poly_add(1:10:end),'-*','Color', cmap(21, :),'LineWidth',lnwd,'MarkerSize',mksz);
hold on;
p21=plot(0:0.005:(2-0.005),sprd_poly_add,'--','Color', cmap(21, :),'LineWidth',lnwd);
hold on;
axis([0 2 0 8]);



hl = legend([p2 p5 p8 p11 p14 p17 p20 p21],...
    'RMSE (XU-sml-w*)', 'RMSE (X-sml-w*)', 'RMSE (XU-sml-w)', 'RMSE (X-sml-w)', ...
    'RMSE (XU-sml-r)', 'RMSE (X-sml-r)','RMSE (poly)', 'Spread',...
    'Location','EastOutside','Orientation','Vertical');

xlabel('Forecast period (MTU)','FontSize',16,'FontWeight','bold');
ylabel('RMSE & Spread','FontSize',16,'FontWeight','bold');
set(hl,'FontSize',32,'FontWeight','bold','LineWidth',lnwd);
set(gca,'FontSize',32,'FontWeight','bold','LineWidth',lnwd);
set(gcf,'Color','w');
%print('-r200','-depsc',['rmse_spread_poly_gans_700_803_exp_20_stoch_Red_newColors.eps']);




