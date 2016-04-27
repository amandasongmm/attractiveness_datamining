% leaveOneOut, test tsne
close all; clear; clc; 

load('cleanFaceFeatureArray.mat');%cleanFaceFeatureArray
faceArray = cleanFaceFeatureArray; %imN * imFeature

targetInd = 6; 
targetArray = faceArray(:,targetInd);
targetArray = round(targetArray);

temp = faceArray; 
temp(:,targetInd) = [];
remainArray = temp; 

%   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
X = remainArray; 
labels = targetArray; 
initial_dims = size(remainArray,2);

mappedX_withoutEmo = tsne(X, labels, 3, initial_dims);

%% 
% handle = figure(100); 
% scatter(mappedX(:,1), mappedX(:,2), 5, labels);
% hold on; 
% legend(handle, '1','2','3','4','5','Location','Best');

%% draw 5 figures separately
% figure(3);
% colorList = cell(5,1);
% colorList{1}='m';colorList{2}='r';colorList{3}='g';colorList{4}='b';colorList{5}='k';
% for curD = 1 : 5
%     indList = labels==curD; 
%     tempMappedX = mappedX(indList,:);
%     scatter(tempMappedX(:,1),tempMappedX(:,2),4,colorList{curD});
%     hold on; 
% end

% %% use all semantic features
% load('cleanFaceFeatureArray.mat');%cleanFaceFeatureArray
% faceArray = cleanFaceFeatureArray; %imN * imFeature
% 
% targetInd = 1; 
% targetArray = faceArray(:,targetInd);
% targetArray = round(targetArray);
% 
% % temp = faceArray; 
% % temp(:,targetInd) = [];
% % remainArray = temp; 
% 
% %   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
% X = faceArray; 
% labels = targetArray; 
% initial_dims = size(faceArray,2);
% 
% mappedX = tsne(X, labels, 2, initial_dims);