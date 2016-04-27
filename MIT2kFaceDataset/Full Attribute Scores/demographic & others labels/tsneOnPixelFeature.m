% tsneOnPixelFeature.m
%load('resizedImData.mat');
load('cleanFaceFeatureArray.mat');%cleanFaceFeatureArray

%featureNum = size(cleanFaceFeatureArray,2);
labelArray = round(cleanFaceFeatureArray);
%1. Age
%2. Attractive
%3. Famous
%4. Common
%5. HowMuchEmo
%6. WhichEmo: disgust, happiness, sadness, anger, fear, surprise, neurtral.
%7. Friendly
%8. Makeup
%9. Gender
%10. Race
%11. Memorable
curInd = 9;
labels = labelArray(:,curInd);
initial_dims = 10; 
mappedX = tsne(resizedImData, labels, 2, initial_dims);
