% correlationCheck.m

% load('cleanFaceFeatureArray.mat');
% 
% %% attractiveness vs friendly
% x = cleanFaceFeatureArray(:,2); % attractive
% y = cleanFaceFeatureArray(:,7); % friendly
% figure; 
% scatter(x,y,3);

%%
clear; clc;close all; 
load('mappedX.mat');
load('cleanFaceFeatureArray.mat');%cleanFaceFeatureArray

featureNum = size(cleanFaceFeatureArray,2);
labelArray = round(cleanFaceFeatureArray);

figure(1); 
label1 = labelArray(:,2);%attractiveness
gscatter(mappedX(:,1),mappedX(:,2),label1);
legend('a1','a2','a3','a4','a5');

figure(2);
label2 = labelArray(:,11);%memorability
gscatter(mappedX(:,1),mappedX(:,2),label2);
legend('m1','m2','m3','m4','m5');