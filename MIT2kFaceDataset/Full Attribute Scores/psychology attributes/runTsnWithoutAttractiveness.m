

load('socialFeatureList.mat');

%% specify parameters for tsne.
% input: all the other social features. 
% label: attractiveness features 1-9. 

temp = socialFeatureList; 
temp(:,16) = [];
X = temp; 
labels = round(socialFeatureList(:,16));
initial_dims = size(X,2);
no_dims = 2; 

mappedX = tsne(X, labels,no_dims,initial_dims);
save('mappedX_withoutAttractiveness.mat','mappedX');
