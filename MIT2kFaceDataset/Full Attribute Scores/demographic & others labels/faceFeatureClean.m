% 
%function cleanedFile = faceFeatureClean(origFile)
load('faceFeatureArray.mat');% 2222*11
cleanFaceFeatureArray = faceFeatureArray(all(~isnan(faceFeatureArray),2),:); % for nan - rows
save('cleanFaceFeatureArray.mat','cleanFaceFeatureArray');
