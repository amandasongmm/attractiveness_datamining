------------------------------
Attribute Scores
------------------------------
Updated December 17, 2013

Several additional pieces of information were collected for the 2,222 target photos from the 10k Face Photograph Database, including:
 - Demographic and other labels (e.g., computer vision and graphics-related questions) on the faces. There are labels from 12 subjects on 19 attributes for all 2,222 target images.
 - A study collecting scores on the target photos using psychology attributes and thesaurus-based antonyms from Oosterhof & Todorov (2008), Vokey & Read (1992), and some attributes in the demographic study that came out as significantly related to memorability. This study got ratings from 30 subjects (15 on the true attribute, 15 on its antonym with a reversed scale) on 20 attributes for all 2,222 target images. These attributes were used in multiple linear regressions in the paper to look at influences of these attributes on memorability.

**Cite this paper when using anything from this dataset:**

Bainbridge, W. A., Isola, P., & Oliva, A. (2013). The Intrinsic Memorability of Face Photographs. Journal of Experimental Psychology: General, 142(4), 1323 - 1334.

------------------------------
Contents:

---
license-agreement.txt
The license agreement attached to this database. You must agree to the terms of this license agreement to use this database.

target-filenames.txt
For this experiment, attributes and labels were collected for 2,222 target photos in a memory game. The target file names from the original database were renamed to number labels (1-2222) for ease of use. This is a text file indicating the number label for each of the 2,222 target images. These number labels are particularly useful for accessing the data in the MATLAB structures.

---
Folder: demographic & others labels
Demographics and Miscellaneous Data
This folder includes data from  a study that collected useful demographic information on each of the photos, as well as labels for various computer vision and graphics applications.

demographic-others-labels.mat
A MATLAB file containing data on 20 demographics and other miscellaneous questions, with 12 participants per face. NaN indicates questions that a participant did not answer. 
 - headers  : The keyword associated with the question for each column in allData (see allData for more detailed description of each question).
 - allData  : The attribute scores for each face. To access a specific face's data, use its number label (#1-2222) as coded in target-filenames.txt as the index to access allData. Each cell contains a 12 x 20 matrix of 12 participants' ratings on the 20 questions for each image. The columns and corresponding possible values are as follows:
  Column 1 - Face's Age
		1 = < 20 years
		2 = 20 - 30 years
		3 = 30 - 45 years
		4 = 45 - 60 years
		5 = 60+ years
  Column 2 - How attractive is this face?
		1 (unattractive) - 5 (attractive)
  Column 3 - Is this person famous?
		0 = No		1 = Maybe		2 = Yes
  Column 4 - How common is this face?
		1 (uncommon) - 5 (common)
  Column 5 - How much emotion is in this face?
		1 (little) - 5 (a lot)
  Column 6 - Emotion?
		0 = Neutral
		1 = Happiness
		2 = Sadness
		3 = Anger
		4 = Fear
		5 = Surprise
		6 = Disgust
 Column 7 - Eyes direction?
		1 = At you
		2 = Up
		3 = Down
		4 = Left (of screen)
		5 = Right (of screen)
 Column 8 - Face direction?
		1 = At you
		2 = Up
		3 = Down
		4 = Left (of screen)
		5 = Right (of screen)
 Column 9 - Facial hair?
		0 = None		1 = A little		2 = A lot
 Column 10 - Catch question
		This was a question to ensure participants were seriously answering the questions. They had to identify the color of a randomly colored frame around the face (red, green, or blue). If this value is 0, then the participant failed the question, and if it is 1, then they passed.
 Column 11 - How friendly is this person?
		1 (very unfriendly - 5 (very friendly)
 Column 12 - Makeup?
		0 = None		1 = A little		2 = A lot
 Column 13 - Gender?
		0 = Female		1 = Male
 Column 14 - Would you cast this person as the star of a movie?
		0 = No			1 = Maybe			2 = Yes
 Column 15 - Would this be a good profile picture?
		0 = No			1 = Maybe			2 = Yes
 Column 16 - Image quality?
		1 (poor) - 5 (very good)
 Column 17 - Race?
		Note: These races were selected based on common demographics of Amazon Mechanical Turk.
		0 = Other
		1 = White
		2 = Black
		3 = East Asian
		4 = South Asain
		5 = Hispanic
		6 = Middle Eastern
 Column 18 - How memorable is this face?
		1 (forgettable) - 5 (memorable)
 Column 19 - At what speed do you think this expression is happening?
		1 (slowly) - 5 (quickly)
 Column 20 - How much teeth is showing?
		0 = None		1 = A little		2 = A lot
 - finalVals : Formatted similarly to allData (i.e., you access the data for each photo using its corresponding number label), except that the cells contain consolidating values across the 12 participants for each attribute. Depending on the question type, this value is either the modal answer for qualitative questions (e.g., race, gender, etc), or the mean answer for quantitative questions (e.g., attractiveness, image quality, etc).
 - stdevVals : Formatted similarly to allData (i.e., you access the data for each photo using its corresponding number label), except that the cells contain values indicating participant agreement across the 12 participants' answers for each attribute. Depending on the question type, this value is either the percent of people in agreement for the modal answer for qualitative questions (e.g., race, gender, etc), or the standard deviation score for quantitative questions (e.g., attractiveness, image quality, etc).
 - operations: A matrix of 20 entries that indicates for each of the 20 columns what calculation was used to determine finalVals and stdevVals. 'a' corresponds to average and standard deviation, while 'm' corresponds to mode and percent agreement.
 
demographic-others-labels.xlsx
Formatted similarly to demographic-others-labels.mat, except it is an Excel spreadsheet that also includes columns in each sheet for the image number label and its original filename.
  See allData for a description of Sheet 1 (All Data) and the possible values for each column. The columns are in the same order for both the .mat and the .xlsx.
  See finalVals for a description of Sheet 2 (Final Values)
  See stdevVals for a description of Sheet 3 (StDev Values)
  Sheet 4 (Operations) lists the operations that were conducted on each column (either average/standard deviation or mode/percent agreement) to determine Final Values and StDev Values.
  
demographic-others-survey.html
An HTML webpage that shows the format of the survey. "face" indicates where the face photograph would be shown. For more details on the implementation of the survey, refer to the original paper.

---
Folder: psychology attributes
Psychology Literature Attributes Data
(Oosterhof & Todorov, 2008; Vokey & Read, 1992)

psychology-attributes.mat
A MATLAB file containing data on the 20 attributes, plus 5 additional fields for additional questions, for the original word and its antonym. The original words and antonyms were randomly split into two different surveys, so the results of the two surveys (uncombined) are contained here. All attributes were rated on a scale of 1 (not at all) - 9 (extremely). NaN indicates questions that a subject did not answer. This dataset also includes participant demographics.
 - headers  : The attribute name corresponding to each column. Attribute names are almost exactly as asked except for some key exceptions:
	* catch & catch ans - this is a catch question (and the participant's answer) meant to ensure participants were carefully reading questions. They were asked to identify a randomly-determined number (1-9) presented on the screen. Catch is the random number, while catch ans is the participant's answer. Data were only used from participants who correctly answered the question (where catch and catch ans were not different).
	* emotStable & emotUnstable - abbreviations for "Emotionally Stable" and "Emotionally Unstable"
	* subage, subrace, submale - participant demographic questions. The participant was asked to only answer once, so most answers will be NaN. Race and age options were determined based on common Amazon Mechanical Turk demographics. The numbers correspond to:
		Age
		1 = < 20 years
		2 = 20 - 30 years
		3 = 30 - 45 years
		4 = 45 - 60 years
		5 = 60+ years
		Male (Gender)
		0 = Female
		1 = Male
		Race
		0 = Other
		1 = White
		2 = Black
		3 = East Asian
		4 = South Asian
		5 = Hispanic
		6 = Middle Eastern
 - allData  : The attribute scores for each face. To access a specific face's data, use its number label (#1-2222) as coded in target-filenames.txt as the index to access allData. Each cell contains a 15 x 50 matrix of 30 participants' ratings on the 50 attributes for each image. Note the 50 attributes are actually 2 surveys of randomly distributed opposite attributes and are combined in the paper itself, so each of the 15 rows in fact contains 2 participants (30 participants total).
 - finalVals : Formatted similarly to allData (i.e., with data for a specific image accessed by its number label), except that the cells contain the mean values across the 15 participants per survey. The values for the catch questions and subject demographics don't mean much here.
 - stdevVals : Formatted similarly to allData (i.e., with data for a specific image accessed by its number label), except that the cells contain the standard deviation values across the 15 participants per survey. The values for the catch questions and subject demographics don't mean much here.
 
psychology-attributes.xlsx
Formatted similarly to attribute-scores.mat, except it is an Excel spreadsheet. Each sheet also has an extra column for the image number label and its original filename.
  See allData for a description of Sheet 1 (All Data). The columns are in the same order for both the .mat and the .xlsx.
  See finalVals for a description of Sheet 2 (Final Values)
  See stdevVals for a description of Sheet 3 (StDev Values)
  
psychology-survey-version1.html & psychology-survey-version2.html
2 HTML files that are the 2 different versions of the survey given to Amazon Mechanical Turk participants. "face" indicates where the photograph would be shown. For details on catch questions and design elements of this survey, please view the original paper.