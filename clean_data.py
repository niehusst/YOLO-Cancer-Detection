import pandas as pd

# A quick script to clean unnecessary columns out of the raw 
# CCC dataset from Kaggle

# read in raw data_label file
CCC = pd.read_csv('CrowdsCureCancer2017Annotations.csv')

# create a column that is the image path for each row
CCC["imgPath"] = CCC['patientID'].map(str) + '/' + \
                CCC['seriesUID'].map(str) + '/' +  \
                CCC['sliceIndex'].map(str) 

# chop out columns we won't need for simple object detector
columns_to_drop = ['order', 'patientID', 'seriesUID', 'sliceIndex', 'StudyTime', 'SOPClassUID', 'instanceUID', 'length','annotator', 'radiologist_status', 'date_unix', 'date', 'StudyInstanceUID', 'StudyDate', 'StudyTime', '_id']

CCC.drop(columns_to_drop, inplace=True, axis=1)

# write csv with clean data
CCC.to_csv('CCC_clean.csv')

