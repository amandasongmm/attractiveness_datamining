% add the legends

% figure(100);
% gscatter(mappedX(:,1), mappedX(:,2), labels);
% legend('Attractive level 2',...
%     'Attractive level 3','Attractive level 4',...
%     'Attractive level 5','Attractive level 6',...
%     'Attractive level 7','Attractive level 8');
% title('Attractiveness clustering','fontSize',16);

%% attractiveness vs interesting
figure(1);
subplot(1,2,1);
curInd = 16; 
labels = round(socialFeatureList(:,curInd));
gscatter(mappedX(:,1), mappedX(:,2), labels);
legend('Attractive level 2',...
    'Attractive level 3','Attractive level 4',...
    'Attractive level 5','Attractive level 6',...
    'Attractive level 7','Attractive level 8');
title('Attractiveness clustering','fontSize',16);

%figure(2);
subplot(1,2,2);
curInd = 22; 
labels = round(socialFeatureList(:,curInd));
gscatter(mappedX(:,1), mappedX(:,2), labels);
legend('Interesting level 2',...
    'Interesting level 3','Interesting level 4',...
    'Interesting level 5','Interesting level 6',...
    'Interesting level 7');
title('Interestingness clustering','fontSize',16);