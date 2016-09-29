clear all; close all; clc;

N = 8000;

for i = 8764:N+2000
    x = spatialPattern([224,224],-1);
    im = imagesc(x);
    axis off;
    fileName = sprintf('~/Documents/pink_noise_image/pink_noise_%d',i);
    set(gca,'position',[0 0 1 1],'units','normalized')
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 2.24 2.24])
    saveas(im,fileName,'png');
    close all;
end