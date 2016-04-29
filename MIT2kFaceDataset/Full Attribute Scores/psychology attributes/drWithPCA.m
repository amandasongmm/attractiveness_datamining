% dr with PCA
load('socialFeatureList.mat');

x = socialFeatureList(:,16);
y = socialFeatureList;%n*23
%y(:,16) = [];

[mappedY] = pca(y,2);


x = round(x);
gscatter(mappedY(:,1), mappedY(:,2),x);
legend('2','3','4','5','6','7','8');