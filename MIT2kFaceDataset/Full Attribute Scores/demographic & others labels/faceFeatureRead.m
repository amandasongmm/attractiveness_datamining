function faceFeatureImport(workbookFile,sheetName,startRow,endRow)
%IMPORTFILE Import data from a spreadsheet

%% Input handling
%Example:
% faceFeatureImport('demographic-others-labels.xlsx','Final Values',2,2223);
%
% If no sheet is specified, read first sheet
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% If row start and end points are not specified, define defaults
if nargin <= 3
    startRow = 2;
    endRow = 2223;
end

%% Import the data
[~, ~, raw] = xlsread(workbookFile, sheetName, sprintf('B%d:V%d',startRow(1),endRow(1)));
for block=2:length(startRow)
    [~, ~, tmpRawBlock] = xlsread(workbookFile, sheetName, sprintf('B%d:V%d',startRow(block),endRow(block)));
    raw = [raw;tmpRawBlock]; %#ok<AGROW>
end
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Allocate imported array to column variable names
Image = data(:,1);
Age = data(:,2);
Attractive1 = data(:,3);
Isthispersonfamous1 = data(:,4);
Common1 = data(:,5);
Howmuchemotionisinthisface1 = data(:,6);
Emotion = data(:,7);
Eyesdirection = data(:,8);
Facedirection = data(:,9);
Facialhair = data(:,10);
Catchquestion = data(:,11);
Friendly = data(:,12);
Makeup = data(:,13);
Gender = data(:,14);
Wouldyoucastthispersonasthestarofamovie = data(:,15);
Wouldthisbeagoodprofilepicture = data(:,16);
Imagequality = data(:,17);
Race = data(:,18);
Memorable = data(:,19);
Atwhatspeeddoyouthinkthisexpressionishappening = data(:,20);
Howmuchteethisshowing = data(:,21);


reserveList = ['Age','Attractive','Famous','Common','HowMuchEmo','WhichEmo','Friendly','Makeup','Gender','Race','Memorable'];
faceFeatureArray = zeros(size(data,1),11);
faceFeatureArray(:,1) = Age; 
faceFeatureArray(:,2) = Attractive1; 
faceFeatureArray(:,3) = Isthispersonfamous1; 
faceFeatureArray(:,4) = Common1; 
faceFeatureArray(:,5) = Howmuchemotionisinthisface1; 
faceFeatureArray(:,6) = Emotion; 
faceFeatureArray(:,7) = Friendly; 
faceFeatureArray(:,8) = Makeup; 
faceFeatureArray(:,9) = Gender; 
faceFeatureArray(:,10) = Race; 
faceFeatureArray(:,11) = Memorable; 
save('faceFeatureArray.mat','faceFeatureArray','reserveList');




