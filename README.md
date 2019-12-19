I first read in the csv files into pandas dataframes for both the train and test sets.

Train set size: 199523
Test set size: 99762

Data cleaning steps:
1. Dropped instance_weight column.
2. Dropped duplicates.
  - There were 46627 duplicated rows in train and 20898 duplicated rows in test.
  - Train set size after dropping duplicates: 145776
  - Test set size after dropping duplicates: 75428
  - Dropping duplicates seems to be a critical step in data cleaning. For example, after dropping duplicates, the mean age increased from 34.49 to 40.05. This suggests that leaving duplicates in for analysis would have skewed results toward the younger population.
3. There were no null values for either train or test.
4. Split income label into binary integer values.

|         | - 50,000       | 50,000+      |
| ------- | -------------- | ------------ |
| Train   | 133424 (91.53%)| 12352 (8.47%)|
| Test | Text |
