# linearRegression_amanda

1. feature_matrix.csv stores the face_num * feature_num feature matrix. 
face_num = 200
feature_num = 24 (but should add 5 more features)

The 200 images are organized in this way: 1-50 MIT, 51-100 Glasgow, 101-150 Genhead, 151-200 ngs(face others)

As it's slightly different from how the rating matrix (matrix_form.csv) is organized, its sequence is changed in the code. 

2. rating_matrix.csv stores the rating of 1548 raters on 200 faces. rater_num * face_num 
rater_num = 1548 (cannot remember how I organized them. need to re-write the code when we need twin-pair index)
face_num = 200. The faces are organized in this way: MIT, Glasgow, face others, genhead

3. convert_feature_matrix_to_new_order.ipynb
   change the image order of the feature_matrix,csv
   make it the same as the rating_matrix.csv

   save the reordered data into new_feature_matrix.csv

4. PCA_analysis.ipynb
   Do PCA_analysis on the new_feature_matrix
   Plot variance vs # of PCs




