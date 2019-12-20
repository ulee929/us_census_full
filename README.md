# US Census Income Analysis

## Data Source:
This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html

Donor:
```
Terran Lane and Ronny Kohavi
Data Mining and Visualization
Silicon Graphics.
e-mail: terran@ecn.purdue.edu, ronnyk@sgi.com for questions.
```
The data was split into train/test in approximately 2/3, 1/3 proportions using MineSet's MIndUtil mineset-to-mlc.

Prediction task is to determine the income level for the person represented by the record. Incomes have been binned at the $50K level to present a binary classification problem.


### Data cleaning steps:
I first read in the csv files into pandas DataFrames for both the train and test sets.

- Train set size: 199523
- Test set size: 99762

I then followed completed the following:
1. Dropped instance_weight column.
2. Dropped duplicates.
  - There were 46627 duplicated rows in train and 20898 duplicated rows in test.
  - **Train set size after dropping duplicates**: 145776
  - **Test set size after dropping duplicates**: 75428
  - Dropping duplicates seems to be a critical step in data cleaning. For example, after dropping duplicates, the mean age increased from 34.49 to 40.05. This suggests that leaving duplicates in for analysis would have skewed results toward the younger population.
3. Dropped detailed_industry_recode and detailed_occupation_recode columns since they are duplicates of major_industry_code and major_occupation_code.
4. There were no null values for either train or test.
5. Put education into buckets so that below 1st grade - 6th grade = up_to_6th_grade and 9th grade - 12th grade no diploma = high_school_no_graduate

| Education level  | Number (train set)  |
| --------------- | ------------- |
|High school graduate                 |     42206|
| Some college but no degree           |     25740|
| Bachelors degree(BA AB BS)            |    19137|
| High school no graduate                 |   18729|
|Children                                 |  9742|
| Masters degree(MA MS MEng MEd MSW MBA)   |  6396|
| 7th and 8th grade                         | 5766|
| Up to 6th grade                      |       5542|
| Associates degree-occup /vocational  |      5173|
| Associates degree-academic program    |     4295|
| Prof school degree (MD DDS DVM LLB JD) |    1789|
| Doctorate degree(PhD EdD)               |   1261|

6. Split income label into binary integer values.

|         | Under $50,000   | Over $50,000  |
| ------- | --------------- | ------------- |
| Train   | 133,424 (91.53%)| 12,352 (8.47%)|
| Test    | 69,242 (91.80%) | 6186 (8.20%)  |




## EDA

### Continuous Variables

**Distribution of continuous variables**
![hist](imgs/hist.png)

**Outlier analysis**
![boxplot](imgs/boxplot.png)
(add corr matrix)
Looks like capital gains, capital losses, dividends, and weeks worked per year are highly correlated with income

outliers:
Replaced outlier values with means for ['wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks'].

#### Categorical Variables
(add visualization)


*Initial results*
accuracy: 0.9278436461738911
precision: 0.6554216793596649
recall: 0.30543324071438405
rmse: 0.26861196824997646
