% myDraft
attractFullArray = attractive(~isnan(attractive));
%hist(attractFullArray);
% cleanFaceFeatureArray = faceFeatureArray(all(~isnan(faceFeatureArray),2),:); % for nan - rows
% save('cleanFaceFeatureArray.mat','cleanFaceFeatureArray');

normList = zeros(9,1);
for curL = 1 : 9
    normList(curL) = (sum(attractFullArray==curL))/length(attractFullArray);
end

