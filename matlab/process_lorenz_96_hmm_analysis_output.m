cmp = colormap('parula');


load lorenz_96_hmm_out.mat





ttl2{1} = 'Truth';
ttl2{2} = 'XU-lrg-w';
ttl2{3} = 'XU-med-w';
ttl2{4} = 'XU-sml-w';
ttl2{5} = 'XU-tny-w';
ttl2{6} = 'X-med-w';
ttl2{7} = 'X-sml-w';
ttl2{8} = 'X-tny-w';
ttl2{9} = 'XU-lrg-r';
ttl2{10}= 'XU-med-r';
ttl2{11}= 'XU-sml-r';
ttl2{12}= 'XU-tny-r';
ttl2{13}= 'X-med-r';
ttl2{14}= 'X-sml-r';
ttl2{15}= 'X-tny-r';
ttl2{16}= 'XU-lrg-w*';
ttl2{17}= 'XU-med-w*';
ttl2{18}= 'XU-sml-w*';
ttl2{19}= 'XU-tny-w*';
ttl2{20}= 'X-sml-w*';
ttl2{21}= 'X-tny-w*';
ttl2{22}= 'Poly';


% set rgs values to ensure consistent association of colour with regime for the 
% different subplots 
rgs = [1 2 2 1 1 1 2 2 2 2 2 2 2 2 2 1 2 1 2 1 1 2];


pltfig1 = 1;

if (pltfig1==1)
figure(1)
for k=1:22
    subplot(8,3,k)
    cont_ref = (0:0.05:1.2)*max(max(f{1,1}));
    cont = cont_ref;
    if (k>0)
        contour(r1{1,1},r2{1,1},f{1,1}',cont_ref,'color',[.7 .7 .7],'linewidth',3)
    end
    hold on
    contour(r1{k,1},r2{k,1},f{k,1}',cont,'k');
    if (rgs(k)==1)
        clr1 = 'r';
        clr2 = 'b';
    else
        clr1 = 'b';
        clr2 = 'r';
    end
    hold on
    if (rgs(k)==1)
        contour(r1{k,1},r2{k,1},g1{k,1}',cont,clr1);
        contour(r1{k,1},r2{k,1},g2{k,1}',cont,clr2);
    else
        contour(r1{k,1},r2{k,1},g2{k,1}',cont,clr2);
        contour(r1{k,1},r2{k,1},g1{k,1}',cont,clr1);
        
    end
    title(ttl2{k})
    if ((k-1)/3==floor((k-1)/3))
        ylabel('Wavenumber 2')
    end
    if (k>12)
        xlabel('Wavenumber 1')
    end
    
    tmp_str = num2str(round(1000*Q{k})/1000);
    
end
orient tall
print -dpdf process_lorenz_96_hmm_output_summary.pdf

end

hmm_path = pth{1};
save hmm_path_out.mat hmm_path

figure(2)

subset = [1 22 4 7 11 14 18 20];

for k=1:8
    
    plt_tmp=subplot(4,4,k);
    cont_ref = (0:0.05:1.2)*max(max(f{1,1}));
    
    cont = cont_ref;
    if (k>0)
        contour(r1{1,1},r2{1,1},f{1,1}',cont_ref,'color',[.7 .7 .7],'linewidth',1)
    end
    hold on
    contour(r1{subset(k),1},r2{subset(k),1},f{subset(k),1}',cont,'k');
    hold on
    if (rgs(subset(k))==1)
        clr1 = 'r';
        clr2 = 'b';
        contour(r1{subset(k),1},r2{subset(k),1},g2{subset(k),1}',cont,clr2);
        contour(r1{subset(k),1},r2{subset(k),1},g1{subset(k),1}',cont,clr1);
        
    else
        clr1 = 'b';
        clr2 = 'r';
        contour(r1{subset(k),1},r2{subset(k),1},g1{subset(k),1}',cont,clr1);
        contour(r1{subset(k),1},r2{subset(k),1},g2{subset(k),1}',cont,clr2);
        
        
    end
    hold off
    
    dy = .02;
    
    if (k<=4)
        set(gca,'xtick',[])
    end
    if (k>4)
        psn = get(plt_tmp,'position');
        set(plt_tmp,'position',psn+[0 dy 0 0])
    end
    
    title(ttl2{subset(k)})
    if ((k-1)/4==floor((k-1)/4))
        
        ylabel('| Z_{2} |')
    end
    if (k>4)
        
        xlabel('| Z_{1} |')
    end
    if (rgs(subset(k))==1)
    tmp_str = num2str(round(1000*Q{subset(k)})/1000);
    else
        Q_tmp = Q{subset(k)};
        Q2(1,1) = Q_tmp(2,2);
        Q2(2,2) = Q_tmp(1,1);
        Q2(1,2) = Q_tmp(2,1);
        Q2(2,1) = Q_tmp(1,2);
        tmp_str = num2str(round(1000*Q2)/1000);
    end
    text(9,17,tmp_str(:,[1:7 12:17]),'fontsize',8)
   
end


print -dpdf lorenz_bivariate_regimes_small.pdf