import numpy as np
import pandas as pd
pd.options.display.max_columns=100

from eda import Visualizations

train_data = pd.read_pickle('df_income_learn.pkl')
test_data = pd.read_pickle('df_income_test.pkl')

df_income_learn = train_data.copy()
df_income_test = test_data.copy()

continuous_vars = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks', 'num_persons_worked_for_employer', 'weeks_worked_in_year', 'income']
df_income_learn_cont = pd.DataFrame([df_income_learn.pop(x) for x in continuous_vars]).T


# Pop the label from dataframes
income_learn_label = df_income_learn.pop('income')
# income_test_label = df_income_test.pop('income')
#
def replace_outliers_with_means(df, column):
    mean = float(df[column].mean())
    df[column] = np.where(df[column] > mean, mean, df[column])
#
# # There are 7 continuous variables
# df_income_learn_cont = pd.DataFrame([df_income_learn.pop(x) for x in continuous_vars]).T
#
# # outlier
# outlier_variables = ['wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks']
for column in outlier_variables:
    replace_outliers_with_means(df_income_learn_cont, column)
#
# # Categorical variables
# df_income_learn_with_dummies = pd.get_dummies(df_income_learn.astype(str))

def kfolds_cv(X, estimator, n_splits=5):
    kfold = KFold(n_splits, shuffle=True)

    accuracies = []
    f1s = []
    rmse = []

    for train_index, test_index in kfold.split(X):
        model.fit(X[train_index], y[train_index])
        y_predict = model.predict(X[test_index])
        y_true = y[test_index]
        rmse_scores = math.sqrt(mse(y_predict, y_true))
        accuracies.append(accuracy_score(y_true, y_predict))
        rmse.append(rmse_scores)
        f1s.append(f1_score(y_true, y_predict))

    print(f"accuracy: {np.average(accuracies)}")
    print(f"f1: {np.average(f1s)}")
    print(f"rmse: {np.average(rmse)}")

if __name__ == '__main__':
    cat_vars = [column for column in df_income_learn.columns]
    Visualizations(df_income_learn, cat_vars).categorical_bars()
