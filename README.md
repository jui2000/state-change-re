# state-change-re

## Enhanced Distant Supervision with State-Change Information forRelation Extraction


## Data

We use the LDC2011T07 dataset and pre-process it, even add manual annotations for our usecase. We provide indexes of the data we have used so that the actual data can be retrived.

You can run the get_original.py file to get the actual sentences, tokens and subject-object names in sentences by giving the dataset directory as input path. 

You can run:

python ./data/get_original.py input_path output_path


In data, there are 5 folders :
1) train contains all the training data with the format - train_windowsize_relationtype.txt which contains all the positives referring to a particular relation. It also contains a train_negatives files which contains 10k fixed negatives which are to be used in addition to the positives for training each sceanrio.
2) val contains all the static validation data for each relation type.
3) test contains all the static test data for each relation type.
4) dynamic_val contains all the dynamic validation data for each relation type.
5) dynamic_test contains all the dynamic test data for each relation type.
