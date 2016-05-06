%% run various linear regression on the data. 


load('socialFeatureList.mat');

y = socialFeatureList(:,16);
temp = socialFeatureList; 
temp(:,16) = [];
X = temp; 

% linear term. robust
%mdl = fitlm(X,y,'linear','RobustOpts','on');
mdl = fitlm(X,y,'interactions','RobustOpts','on');