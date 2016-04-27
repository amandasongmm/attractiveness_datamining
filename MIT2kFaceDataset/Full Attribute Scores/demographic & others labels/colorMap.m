%% the purpose is to check which semantic feature are correlated with each other, 
% which reflects the cognitive structure of those features. 
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




clear; clc; 
load('mappedX.mat');
load('cleanFaceFeatureArray.mat');%cleanFaceFeatureArray

featureNum = size(cleanFaceFeatureArray,2);
labelArray = round(cleanFaceFeatureArray);

for curItr = 6%1 : featureNum
    figure; 
    curLabel = labelArray(:,curItr);
    gscatter(mappedX(:,1), mappedX(:,2), curLabel);
    legend('Disgust','Happiness','Sadness','Anger','Fear','Surprise','Neutral');
end



% yData = mappedX(:,2);
% col = labelArray(:,9);
% figure;
% gscatter(xData, yData, col);
% legend('Male','Female');









%figure; scatter(mappedX(:,1), mappedX(:,2), 5, labels); 