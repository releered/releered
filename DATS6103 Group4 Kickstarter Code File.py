#%% [markdown]
## Predicting Kickstarter Campaign Success
# By: Leshauna Hartman, Fardin Hafiz, Tanya Visser, and Rachel Thomas (Group 4)

#%% [markdown]

## 1. Data Cleaning
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.linear_model import LogisticRegression
from statsmodels.formula.api import glm
from sklearn.metrics import roc_auc_score, roc_curve
import time
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel
from sklearn.model_selection import validation_curve
from sklearn import metrics
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#%% [markdown]

# We started by reading in the dataset and calling it "kickstarter". 
# Then we ran some code just to evaluate what the dataset looked like
# from a wholistic standpoint - what kind of values were in the dataset
# and the data types and distriution of null values.

#%%
# read in the dataset 

kickstarter = pd.read_csv("ks-projects-201801.csv")

print("\nReady to continue.")
# %%
# first 5 rows 
kickstarter.head()
# %%
# check datatypes 
kickstarter.info()

# check unique values for 'state' and 'main_category'
kickstarter['state'].unique()
kickstarter['main_category'].unique()

#%% [markdown]
# The dataset showed 6 final possible states for a campaign. Since we
# were only interested in whether a campaign succeeded or failed, we 
# created a subset that only contained rows reflecting these outcomes
# (called "kickstarter1"). We then calculated the duration of the campaign
# by finding the difference between the date launched and the deadline 
# date. We changed the relevant variable from the default object
# data type to categorical. Finally we created a subset called "kickstarter_final"
# containing only the variables we were going to consider (main_category, currency, state, backers, country,
# usd_pledged_real, usd_goal_real, Duration). The subcategories would have
# likely introduced multicollinearity with the main_categories, and it make
# more sense to evaluate the funding goals and pledges converted to all 
# US dollars since there were many currency types included in the dataset.

# %%

# subset for just failed or success, reduces set to 331675 rows with 15 variables

kickstarter1 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]
print(kickstarter1)

# %%

#add duration of campain (difference between launch date and deadline)
kickstarter1['launched'] = pd.to_datetime(kickstarter1['launched']).dt.date

# Convert 'deadline' to datetime and then to date (date-only)
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline']).dt.date

# Calculate the duration (difference in days) between deadline and launched
kickstarter1['Duration'] = (pd.to_datetime(kickstarter1['deadline']) - pd.to_datetime(kickstarter1['launched'])).dt.days

# Display the DataFrame
print(kickstarter1[['launched', 'deadline', 'Duration']])
# %%

# change the objects to factors
kickstarter1['main_category'] = kickstarter1['main_category'].astype('category')
kickstarter1['currency'] = kickstarter1['currency'].astype('category')
kickstarter1['state'] = kickstarter1['state'].astype('category')
kickstarter1['country'] = kickstarter1['country'].astype('category')

kickstarter1.info()
# %%
# subset with just main_category, currency, state, backers, country,
# usd_pledged_real, usd_goal_real, Duration

kickstarter_final = kickstarter1[['main_category', 'currency', 'state', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real', 'Duration']]
print(kickstarter_final)

#%% [markdown]

## 2. Exploratory Data Analysis

# The exploratory Data Analysis aimed to understand the shape of the dataset
# and guide the modeling process. This included summary statistics and 
# multiple graph types to understand the relationships between the variables.

# %%
# summary stats (for all countries)
# Describe continuous variables
print(kickstarter_final[['backers', 'usd_goal_real', 'usd_pledged_real', 'Duration']].describe())

# Describe categorical variables
print(kickstarter_final[['main_category', 'state', 'currency', 'country']].apply(lambda x: x.describe(include='all')).T)

#%% [markdown]
# A quick look at the summary statistics shows that number of backers, goal amount, and pledged amount have large ranges with most of the values on the lower range and some extreme values on the higher end. We can also see that Film and Video projects are more prevalent on Kickstarter, the majority of campaigns fail, and the US produces the most campaigns. 

#%%
# distribution of failed vs success - more likely to fail than succeed.

distribution = kickstarter_final['state'].value_counts()
print(distribution)
percentage_distribution = kickstarter_final['state'].value_counts(normalize=True) * 100
print(percentage_distribution)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
kickstarter_final['state'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution of Success vs Failure')
plt.xticks(rotation = 0)
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.pie(
    distribution, 
    labels=distribution.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['red', 'green'])
plt.title('Distribution of Kickstarter Project Outcomes')
plt.show()

# %%

# state by country, currency, and category
grouped_country = kickstarter_final.groupby(['country', 'state']).size().unstack()

grouped_country.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Country, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Country')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

grouped_currency = kickstarter_final.groupby(['currency', 'state']).size().unstack()

grouped_currency.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Currency, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Currency')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

grouped_category = kickstarter_final.groupby(['main_category', 'state']).size().unstack(fill_value=0)

grouped_category.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Category, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Category')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
#%% [markdown]
# In the first 2 graphs above, you can see the total number of projects in the 
# US far exceed that of any other country. In the third, you can the overall
# distribution of projects in the various categories, as well as failure
# versus success rate for those categories. some have appreciably better rates
# of success like dance and theater, while other have appreciably worse rates
# of success like journalism and technology. 

#%% 
# top 5 categories (based on percentage of successful projects in the category)

total_projects_per_category = kickstarter_final.groupby('main_category').size()
successful_projects_per_category = kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage = (successful_projects_per_category / total_projects_per_category) * 100

# Get the top categories with the highest percentage of successful projects
top_categories_percentage = success_percentage.sort_values(ascending=False).head(5)

print("Top 5 categories with the Highest Percentage of Successful Projects (all countries)")
print(top_categories_percentage)

#%% 
# using median goal instead of mean goal

# Calculate the total number of projects per category
total_projects_per_category_all = kickstarter_final.groupby('main_category').size()

# Calculate the total number of successful projects per category
successful_projects_per_category_all = kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage_all = (successful_projects_per_category_all / total_projects_per_category_all) * 100

# Sort the categories by their success percentages
sorted_success_percentage_all = success_percentage_all.sort_values(ascending=False)

print("Percentage of Successful Projects per Category")
print(sorted_success_percentage_all)

# Median goal per category 
median_goal_per_category = kickstarter_final.groupby('main_category')['usd_goal_real'].median()

# Normalize funding goals for color mapping
norm = mcolors.Normalize(vmin=median_goal_per_category.min(), vmax=median_goal_per_category.max())
colors = plt.cm.coolwarm(norm(median_goal_per_category[sorted_success_percentage_all.index]))

fig, ax = plt.subplots(figsize=(12, 8))
sorted_success_percentage_all.plot(kind='bar', color=colors, ax=ax)
plt.title('Percentage of Successful Projects in Each Category (all countries included)', fontsize=14)
plt.ylabel('Percent of Successful Projects (%)')
plt.xlabel('Main Category')
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adding the values onto each bar
for index, value in enumerate(sorted_success_percentage_all):
    ax.text(index, value + 0.5, f'{round(value, 2)}%', ha='center', va='bottom', fontsize=10)

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Goal (USD)')

plt.tight_layout()
plt.show()

#%% [markdown]

# The percentage of successful projects for all of the categories is shown in the above graph. The bars for each category are colored by the median funding goal amount for that category. Generally, the more successful categories had lower funding goals and categories with higher goals were less successful, with technology having the highest median goal and lowest success rate.

#%% 
# top 5 categories percentage (US only)

us_kickstarter_final = kickstarter_final[kickstarter_final['country'] == 'US']

total_projects_per_category_us = us_kickstarter_final.groupby('main_category').size()
successful_projects_per_category_us = us_kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage_us = (successful_projects_per_category_us / total_projects_per_category_us) * 100

# Get the top categories with the highest percentage of successful projects
top_categories_percentage_us = success_percentage_us.sort_values(ascending=False).head(5)

print("Top 5 categories with the Highest Percentage of Successful Projects (all countries)")
print(top_categories_percentage_us)

# using median goal instead of mean goal

# Calculate the total number of projects per category
total_projects_us = us_kickstarter_final.groupby('main_category').size()

# Calculate the total number of successful projects per category
successful_projects_us = us_kickstarter_final[us_kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
successful_projects_us = (successful_projects_us / total_projects_us) * 100

# Sort the categories by their success percentages
sorted_success_percentage_us = successful_projects_us.sort_values(ascending=False)

print("Percentage of Successful Projects per Category")
print(sorted_success_percentage_us)

# Median goal per category 
median_goal_per_category_us = us_kickstarter_final.groupby('main_category')['usd_goal_real'].median()

# Normalize funding goals for color mapping
norm = mcolors.Normalize(vmin=median_goal_per_category_us.min(), vmax=median_goal_per_category_us.max())
colors = plt.cm.coolwarm(norm(median_goal_per_category_us[sorted_success_percentage_us.index]))

fig, ax = plt.subplots(figsize=(12, 8))
sorted_success_percentage_us.plot(kind='bar', color=colors, ax=ax)
plt.title('Percentage of Successful Projects in Each Category (US only)', fontsize=14)
plt.ylabel('Percent of Successful Projects (%)')
plt.xlabel('Main Category')
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adding the values onto each bar
for index, value in enumerate(sorted_success_percentage_us):
    ax.text(index, value + 0.5, f'{round(value, 2)}%', ha='center', va='bottom', fontsize=10)

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Goal (USD)')

plt.tight_layout()
plt.show()

#%% [markdown]

# The percentage of successful projects for all of the categories in the US is shown in the above graph. The bars for each category are colored by the median funding goal amount for that category. Generally, the more successful categories had lower funding goals and categories with higher goals were less successful, with technology having the highest median goal and being on the lower end of success rates. 

#%%
#success and failure by backers and funding goal

plt.figure(figsize=(6, 10))  # Larger plot for better visibility

sns.scatterplot(
    data=kickstarter_final,
    x='state',
    y='backers',
    size='usd_goal_real',  
    hue='usd_goal_real',  
    palette='viridis',  
    sizes=(50, 500),  
    alpha=0.7  
)

plt.title('State by Number of Backers and Funding Goal', fontsize=20)
plt.xlabel('Campaign State', fontsize=14)
plt.ylabel('Number of Backers', fontsize=14)
plt.yscale('log')  # Optional: Use log scale if needed
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Goal (USD)', fontsize=12, title_fontsize=14, loc='center')
plt.tight_layout()
plt.show()

#%% [markdown]
# The graph above shows failed campaigns have fewer backers than successful
# campaigns, and also have significantly larger funding goals than successful
# campaigns indicaged by the colored bubbles on the failed side of the plot. 
# Successful campaigns have more backers and more modest funding goals.

#%%

# Boxplot showing distribution of goal amounts 
plt.figure(figsize=(12, 8))
sns.boxplot(data=kickstarter_final, x='main_category', y='usd_goal_real', palette='coolwarm')
plt.title('Goal Amounts by Category', fontsize = 14)
plt.ylabel('Goal Amount (USD)', fontsize = 12)
plt.xlabel('Main Category', fontsize = 12)
plt.yscale('log')  # To handle wide ranges
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% [markdown]
# The boxplot above shows the distibution of the goal amount by each category
# In this plot, there is a general trend among all the categories with most goals being between 1000 and 10000, with quite a number of projects in each category still exceeding this range.


# %%
# Average % met and shows the inconsistency in the data with the canceled state
# This code aims to find the % of the goal that is being met and also looks to study more of the canceled state of the data.
# This is done by calculating the average of the money raised / total goal. This code does this for the failed and successful 
# data. This code also looks at the number of rows removed that are labeled as "canceled" but are successful since
# they met their goal. 

data = kickstarter[kickstarter['state'].isin(['failed', 'canceled', 'successful'])]

canceled_inconsistent = data[(data['state'] == 'canceled') & (data['usd_pledged_real'] > data['usd_goal_real'])]
num_removed_canceled = canceled_inconsistent.shape[0]

total_canceled = data[data['state'] == 'canceled'].shape[0]
num_kept_canceled = total_canceled - num_removed_canceled

print(f"Number of logically inconsistent 'canceled' rows removed: {num_removed_canceled}")

data_clean = data[~((data['state'] == 'canceled') & (data['usd_pledged_real'] > data['usd_goal_real']))]
data_clean['percentage_met'] = (data_clean['usd_pledged_real'] / data_clean['usd_goal_real']) * 100

data_filtered = data_clean[data_clean['state'].isin(['failed', 'successful'])]

goal_met_percentage = data_filtered.groupby('state')['percentage_met'].mean().reset_index()

plt.figure(figsize=(8, 6))
plt.bar(goal_met_percentage['state'], goal_met_percentage['percentage_met'], color=['red', 'green'])
plt.title('Average Percentange Met of Campaign Goal')
plt.ylabel('Average Percentage of Goal Met (%)')
plt.xlabel('Project State')
plt.ylim(0, max(goal_met_percentage['percentage_met']) + 20)
for index, value in enumerate(goal_met_percentage['percentage_met']):
    plt.text(index, value + 2, f"{value:.2f}%", ha='center')
plt.show()

plt.figure(figsize=(6, 6))
plt.pie([num_removed_canceled, num_kept_canceled],
        labels=['Removed Canceled Rows', 'Kept Canceled Rows'],
        autopct='%1.1f%%', startangle=90, colors=['red', 'orange'])
plt.title('Breakdown of Removed vs. Kept Canceled Rows')
plt.show()

#%%
# Zero backers
# This code aims to find the number of projects with zero backers and compare it to 
# the total number of projects that have at least one backer. This was done by 
# coding to search the count for 'backers' = 0, which would return all of the project
# that had no backers. This was then compared to the projects with at least 1 backer in
# a pie chart.

zero_backers_count = kickstarter[kickstarter['backers'] == 0].shape[0]
one_or_more_backers_count = kickstarter[kickstarter['backers'] >= 1].shape[0]

labels = ['Zero Backers', '1+ Backers']
sizes = [zero_backers_count, one_or_more_backers_count]
colors = ['red', 'green']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Projects: Zero Backers vs 1+ Backers')
plt.show()


# %%
# Histogram of Backers
# To continue the study of the number of backers, a histogram was created in order 
# to study the overall numbers of backers. The outliers were removed from this 
# graph because when they were included, the graph was not very useful since most 
# of the data ended up in one column. We removed the outliers and the graph actually 
# made sense. The libraries were used to make the histogram, and we also made a code 
# to see the number of outliers removed.
backers = kickstarter['backers']

Q1 = backers.quantile(0.25)
Q3 = backers.quantile(0.75)
IQR = Q3 - Q1
filtered_backers = backers[(backers <= Q3 + 1.5 * IQR)]

num_outliers_removed = len(backers) - len(filtered_backers)

plt.hist(filtered_backers, bins=50, edgecolor='black')
plt.title('Histogram of Backers (Outliers Removed)')
plt.xlabel('Number of Backers')
plt.ylabel('Frequency')
plt.show()

print(f"Number of outliers removed: {num_outliers_removed}")

# %%
# Number of One Donor Successful Campaigns 
# This code aims to determine the total number of successful campaigns with only one donor.
# This was done by setting certain factors in the backers' data set like setting the state
# to successful and the backers to one. We then found the number of data that fit these 
# specifc factors and also took the mean pledged of these backers.

successful_one_backer = kickstarter[(kickstarter['state'] == 'successful') & (kickstarter['backers'] == 1)]
average_amount = successful_one_backer['usd_pledged_real'].mean()

print(f"Number of One Doner Success Campaigns: {successful_one_backer.shape[0]}")
print(f"Average amount pledged for successful projects with exactly 1 backer: ${average_amount:.2f}")


#%% [markdown]

## 3. Classification Decision Tree Training, Fit, Analysis, and Evaluation
# To create the decision tree, we created a training set and then fit a 
# tree to max depth 8. This produced a complex tree with 34 leaf nodes, but
# training error rate of 0. The test error rate for this tree was 5.2% which
# is good, however the size of the tree makes it very complex to understand
# and we strongly suspct this tree is overfitting the data. 
# We then tried trees at depths 3. 4. and 5 (only 4 is depicted in this code).
# The tree at depth 3 was reasonably easy to undersand and did still have
# a training error rate of only 6.9%, and a test error rate of 10.6%. At
# max depth 4, the training error rate was 6.87% and the test error rate
# was 8.4%. At max depth 5, the training error was only 1.6% and the test
# error 6.4%. In order to compare the models at the different depths, we 
# completed cross validation at each and compared the scores using T-tests.
# The model with max depth 4 performed statistically better than the model
# with max depth 3, but the model with max depth 5 was not statistically
# better than the model with max depth 4. We also performed a validation
# curve for the tree which showed the ideal depth is probably around 5-6
# based on the point at which both the training score and the validation 
# score are at their highest, however these models at these depths are 
# very complex and much more difficult to understand. So we elected to 
# use the model with max depth 4 as we felt the slight increase error 
# rate was acceptable in exchange for a much simpler model with only 15 
# terminal leaf nodes.
# %%
# create a training set

train_set, test_set = train_test_split(kickstarter_final, train_size=800, random_state=42)

#%%
#fit tree to training data
# max depth 8
X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter8 = DecisionTreeClassifier(max_depth = 8, criterion = 'gini', random_state = 1)

dtree_kickstarter8.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter8.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

# max depth 4
X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter4 = DecisionTreeClassifier(max_depth = 4, criterion = 'gini', random_state = 1)

dtree_kickstarter4.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter4.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

#%%
# Comparison of cross validations to find best depth

X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

# Cross-validation for max_depth=3
dtree_depth3 = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=1)
scores_depth3 = cross_val_score(dtree_depth3, X_trainkickstarter, y_trainkickstarter, cv=5)

# Cross-validation for max_depth=4
dtree_depth4 = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=1)
scores_depth4 = cross_val_score(dtree_depth4, X_trainkickstarter, y_trainkickstarter, cv=5)

dtree_depth5 = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=1)
scores_depth5 = cross_val_score(dtree_depth5, X_trainkickstarter, y_trainkickstarter, cv=5)

t_stat, p_value = ttest_rel(scores_depth3, scores_depth4)

print(f"Scores for max_depth=3: {scores_depth3}")
print(f"Scores for max_depth=4: {scores_depth4}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Check if the difference is significant
if p_value < 0.05:
    print("The difference in performance is statistically significant.")
else:
    print("The difference in performance is not statistically significant.")

# Perform paired t-test
t_stat, p_value = ttest_rel(scores_depth4, scores_depth5)

print(f"Scores for max_depth=4: {scores_depth4}")
print(f"Scores for max_depth=5: {scores_depth5}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Check if the difference is significant 
if p_value < 0.05:
    print("The difference in performance is statistically significant.")
else:
    print("The difference in performance is not statistically significant.")

#%%

# Validation curve for decision tree - probably best around 6, but more complex than necessary

# Define parameter range
param_range = np.arange(1, 15)

# Calculate training and validation scores
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=1),
    X_trainkickstarter, y_trainkickstarter,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray", alpha=0.2)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gray", alpha=0.2)

plt.title("Validation Curve with Decision Tree")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()


#%%

#plot the tree
# max depth 8
plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter8, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter8.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes (max depth 8)')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter8.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")


# max depth 4
plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter4, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter4.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes (max depth 4)')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter4.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")

#%% [markdown]
# The tree with max depth 8 is extremely complex, with 34 terminal leaf nodes.
# The tree with max depth 4 is significanlty less complex, with only 15 terminal leaf nodes.
#%%

# Generate a text summary of the tree
tree_rules8 = export_text(dtree_kickstarter8, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules8)

tree_rules4 = export_text(dtree_kickstarter4, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules4)

#%% [markdown]

# In order to interpret the tree, we generated a text summary which is 
# significanlty more readable than the tree plot. This tree splits
# first by backers, then by goal, then by pledged amount, and then category,
# with a small influence of country. 
# Path 1 has <= 12.5 backers, a goal <= $650, pledge amount <= $184.81, and not in Music are likely to fail.
# Path 2 has <= 12.5 backers, a goal <= $650, pledge amount > #184.81, and not in Publishing are likely to succeed.
# Path 3 has <= 12.5 backers, a goal > $650, having a specific country, and in Dance will likely fail.
# Path 4 has >12.5 backers but <= 67.5, a goal <= $4747, and any value pledged is likely to be successful.
# Path 5 has > 12.5 backers but <= 67.5, a goal is > $4747, and pledged <= $6322.9 are likely to fail, but > $6322.9 are likely to succeed.
# Path 6 has > 67.5 backers, a goal <= $36970.10, and not Crafts are likely to succeed and if Crafts is likely to fail.
# Path 7 has >67.5 backers, a goal > $36970, and pledged amount <= $38512.01 are like to fail, but if > $38512.01 are likely to be successful.
# Overall, backers are the most significant predictor. Projects with fewer than 12.5 backers are most likely to 
# fail regardless of other factors. Projects with backers between 12.5-67.5
# backers increased the likelihood of success as long a goals re small to moderate.
# Projects with > 67.5 backers have the highest likelihood of success, even with higher funding goals.
# Small funding goals succeed more often, with projects with a goal less than 
# $4747 (and especially less than $650), are highly likely to succeed assuming
# they get some backers and some pledged amount. Low pledged amounts leads to 
# failure, especially for high goals. Categories play an overall secondary
# role, though funding for Craft projects is likely to fail in most scenarios.
# There may be some small regional effects, but they are not substantial.

#%%

#predict response on the test data and produce confusion matrix
# max depth 8
X_testkickstarter = pd.get_dummies(test_set.drop(columns=['state']), drop_first=True)

# Align test set columns with training set columns
X_testkickstarter = X_testkickstarter.reindex(columns=X_trainkickstarter.columns, fill_value=0)

y_testkickstarter = test_set['state']


y_testkickstarter_pred = dtree_kickstarter8.predict(X_testkickstarter)

conf_matrix = confusion_matrix(y_testkickstarter, y_testkickstarter_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dtree_kickstarter8.classes_, yticklabels=dtree_kickstarter.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Set (max depth 8)')
plt.show()

test_error_rate = 1 - accuracy_score(y_testkickstarter, y_testkickstarter_pred)
print(f"Test error rate: {test_error_rate:.4f}")

# max depth 4
X_testkickstarter = pd.get_dummies(test_set.drop(columns=['state']), drop_first=True)

# Align test set columns with training set columns
X_testkickstarter = X_testkickstarter.reindex(columns=X_trainkickstarter.columns, fill_value=0)

y_testkickstarter = test_set['state']


y_testkickstarter_pred = dtree_kickstarter4.predict(X_testkickstarter)

conf_matrix = confusion_matrix(y_testkickstarter, y_testkickstarter_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dtree_kickstarter4.classes_, yticklabels=dtree_kickstarter.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Set (max depth 4)')
plt.show()

test_error_rate = 1 - accuracy_score(y_testkickstarter, y_testkickstarter_pred)
print(f"Test error rate: {test_error_rate:.4f}")
# %%
# checking for best depth - although better fits a higher depth, tree becomes very complex and potentially overfit
maxlevels = [None, 2, 3, 4, 5, 6, 7, 8]
crits = ['gini', 'entropy']
for l in maxlevels:
    for c in crits:
        dt = DecisionTreeClassifier(max_depth = l, criterion = c)
        dt.fit(X_trainkickstarter, y_trainkickstarter)
        print(l, c, dt.score(X_testkickstarter, y_testkickstarter))

#%% [markdown]

## 4. Logistic Regression Model Training, Fit, Analysis, and Evaluation

# %%

# Logistic Regression Model
# Features selected: backers, usd_pledged_real, main_category

X = pd.get_dummies(kickstarter_final[['backers', 'usd_pledged_real', 'main_category']], drop_first=True)
y = (kickstarter_final['state'] == 'successful').astype(int) 

# Train-Test Split; 70:30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 251)

# Model is fitted and the accuracy of both sets are evaluated
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Accuracy
print('Logit model accuracy (test set):', model.score(X_test, y_test))
print('Logit model accuracy (train set):', model.score(X_train, y_train))

# Coefficients and Odds Ratios for features
coefficients = pd.DataFrame({
    'Predictors': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})
print("\nCoefficients and Odds Ratios:\n", coefficients)

# Predictions and Evaluation
y_pred = model.predict(X_test)

# Obtaining confusion matrix and accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

#%%
# Printing confusion matrix for logit model
print("\n The confusion matrix of the model is:")
print(conf_matrix)

# Printing accuracy
print("\n The accuracy of the model is:")
print(accuracy)

# Printing classification report
print("\n The model's classification Report:")
print(classification_report(y_test, y_pred))

#%%
TN, FP, FN, TP = conf_matrix.ravel() #Obtaining values from confusion matrix

# Calculating FPR and FNR
FPR = FP / (FP + TN)  # False Positive Rate
FNR = FN / (FN + TP)  # False Negative Rate

# Printing FPR and FNR
print(f"False Positive Rate (FPR): {FPR*100:.2f}%")
print(f"False Negative Rate (FNR): {FNR*100:.2f}%")


#%%
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False, annot_kws={"size": 14}) # add color, size, etc
plt.title("Confusion Matrix", fontsize = 14)
plt.xlabel("Predicted", fontsize = 12)
plt.ylabel("True", fontsize = 12)
plt.show()

#%%

y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
roc_auc = roc_auc_score(y_test, y_prob)  # AUC score

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange', lw=2)
plt.fill_between(fpr, tpr, color='lightcoral', alpha=0.5)  # Filling in AUC by shading
plt.plot([0,1],[0,1],color='black', linestyle='--', lw=1, alpha = 0.7)  # Diagonal line

# Customize the plot
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(fontsize=12, loc=[0.5, 0.1])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%% [markdown]

# The threshold of the original logit model was adjusted as the model had a much higher FNR than FPR.
# This indicated that the model has difficulty identifying successful projects.
# To address this issue, the threshold is lowered.

#%% 
# New Model, adjusted threshold
threshold = 0.3 # threshold adjusted to 0.3 instead of default 0.5
y_pred_adjusted = (y_prob >= threshold).astype(int)

# Confusion Matrix and Metrics with adjusted threshold
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix with Threshold 0.3:\n", conf_matrix_adjusted)

accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
print(f"\nAccuracy with Threshold 0.3: {accuracy_adjusted:.2f}")

print("\nClassification Report with Threshold 0.3:")
print(classification_report(y_test, y_pred_adjusted))

#%%
#Confusion Matrix with Threshold of 0.3:

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_adjusted, annot=True, fmt='d', cmap='viridis', cbar=False, annot_kws={"size": 14})
plt.title("Confusion Matrix with threshold 0.3", fontsize = 14)
plt.xlabel("Predicted", fontsize = 12)
plt.ylabel("True", fontsize = 12)
plt.show()

#%%
new_TN, new_FP, new_FN, new_TP = conf_matrix_adjusted.ravel() # obtaining TP, TN, FP, FN from newly created matrix (i.e. at threshold 0.3)

# Calculating new FPR and FNR at threshold = 0.3
new_FPR = new_FP / (new_FP + new_TN)  # False Positive Rate
new_FNR = new_FN / (new_FN + new_TP)  # False Negative Rate

# Print FPR and FNR at threshold 0.3
print(f"False Positive Rate (FPR) at threshold 0.3: {new_FPR*100:.2f}%")
print(f"False Negative Rate (FNR) at threshold 0.3: {new_FNR*100:.2f}%")


#%% [markdown]

## 5. Logistic Regression Model and Forward Step-wise Feature Selection with the same variables but data subset to US only. 

# %%
# prep US only stuff 

# Create final dataframe with selected columns
kickstarter_final_US = kickstarter_final.copy()

#subset out US 
kickstarter_final_US = kickstarter_final_US[kickstarter_final_US['country'] == 'US']

#rename film category 
kickstarter_final_US['main_category'] = kickstarter_final_US['main_category'].replace({'Film & Video': 'Film_and_Video'})

kickstarter_final_US = kickstarter_final_US.drop(['currency', 'country'], axis = 1)

#%%

#binary state
kickstarter_us_binary = kickstarter_final_US.copy()
kickstarter_us_binary['state_binary'] = kickstarter_us_binary['state'].map({'failed': 0, 'successful': 1})
kickstarter_us_binary['state_binary'] = kickstarter_us_binary['state_binary'].astype(int)

kickstarter_us_binary = kickstarter_us_binary.drop(['state'], axis = 1)

# Dummy variables
kickstarter_us_binary = pd.get_dummies(kickstarter_us_binary, columns=['main_category'], drop_first=True)

# Sample a smaller subset (e.g., 5% of the data) for faster feature selection
def sample_data(df, frac=0.05, random_state=42):
    return kickstarter_us_binary.sample(frac=frac, random_state=random_state)

# Helper function to print timings
def print_timing(message, start_time):
    print(f'{message}: {time.time() - start_time:.2f} seconds')

# Prepare the data
train_df_select, test_df_select = train_test_split(kickstarter_us_binary, test_size=0.2, random_state=42)
x_us_select = train_df_select.drop(columns=['state_binary'], axis=1)
y_us_select = train_df_select['state_binary']

# Sample 5% of the training data for feature selection
sampled_train_df = sample_data(train_df_select, frac=0.05)
x_sampled_us_select = sampled_train_df.drop(columns=['state_binary'], axis=1)
y_sampled_us_select = sampled_train_df['state_binary']

logistic_model = LogisticRegression(max_iter=5000)

print("Starting feature selection...")
start_time = time.time()

# Perform forward feature selection on the sampled data with reduced cross-validation folds
sfs_us = SFS(logistic_model,
             k_features='best',
             forward=True,
             floating=False,
             scoring='accuracy',
             cv=3,  # Reduced number of CV folds
             n_jobs=-1)  # Use all available cores
sfs_us = sfs_us.fit(x_sampled_us_select, y_sampled_us_select)

selection_time = time.time() - start_time
print_timing("Feature selection time", start_time)
selected_features_us = list(sfs_us.k_feature_names_)
print(f'Selected features: {selected_features_us}')

print("Starting model fitting with full data...")
start_time = time.time()
x_selected_us_features = x_us_select[selected_features_us]
logistic_model.fit(x_selected_us_features, y_us_select)
fitting_time = time.time() - start_time
print_timing("Model fitting time", start_time)

print("Starting model evaluation...")
start_time = time.time()
x_test_us_select = test_df_select[selected_features_us]
y_test_us_select = test_df_select['state_binary']
y_pred_us_select = logistic_model.predict(x_test_us_select)
accuracy_us_select = accuracy_score(y_test_us_select, y_pred_us_select)
print(f'Accuracy: {accuracy_us_select}')
evaluation_time = time.time() - start_time
print_timing("Model evaluation time", start_time)



# Convert boolean columns to integers
def convert_boolean_to_int(data):
    bool_cols = data.select_dtypes(include=['bool']).columns
    data[bool_cols] = data[bool_cols].astype(int)
    return data

# Calculating VIF with NaN, Infinity checks, and boolean conversion
def calculate_vif(data):
    # Convert boolean columns to integers
    data = convert_boolean_to_int(data)

    # Convert data to numeric and handle errors
    data = data.apply(pd.to_numeric, errors='coerce')

    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data

# Running VIF on the selected features
x_selected_us_features_with_vif = x_us_select[selected_features_us]
vif_df = calculate_vif(x_selected_us_features_with_vif)
print(vif_df)

#%% [markdown]
# Looking at the above code, we tried forward step-wise feature selection for dataset subset for US country only which chose backers, pledged amount, goal amount, duration, and comics category as the more important variables. The accuracy of the model fit with those variables was high at 0.998 which is almost perfectly predictive. We felt that this may be due to overfitting of the data. So we tried looking at different variations and found that the overfitting occurred when using backers and goal amount. 

# %%
#try us only 
train_df_stats, test_df_stats = train_test_split(kickstarter_us_binary, test_size=0.2, random_state=42)

# Define the formula for logistic regression
formula = 'state_binary ~ backers + usd_pledged_real + main_category_Comics + main_category_Crafts + main_category_Dance + main_category_Design + main_category_Fashion + main_category_Film_and_Video + main_category_Food + main_category_Games + main_category_Journalism + main_category_Music + main_category_Photography + main_category_Publishing + main_category_Technology + main_category_Theater'

# Fit the logistic regression model
stats_model_us = smf.logit(formula, data=train_df_stats).fit()

# Make predictions on the test data
test_df_stats['predicted'] = stats_model_us.predict(test_df_stats)
test_df_stats['predicted_class'] = (test_df_stats['predicted'] > 0.5).astype(int)

# Display the model summary
print(stats_model_us.summary())

# Extract the independent variables from the training data
independent_vars_for_vif = train_df_stats[['backers', 'usd_pledged_real', 'main_category_Comics', 'main_category_Crafts', 
                                   'main_category_Dance', 'main_category_Design', 'main_category_Fashion', 
                                   'main_category_Film_and_Video', 'main_category_Food', 'main_category_Games', 
                                   'main_category_Journalism', 'main_category_Music', 'main_category_Photography', 
                                   'main_category_Publishing', 'main_category_Technology', 'main_category_Theater']]

# Convert boolean variables to integers
independent_vars_for_vif = independent_vars_for_vif.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Ensure all data types are numeric
independent_vars_for_vif = independent_vars_for_vif.apply(pd.to_numeric, errors='coerce')

# Create a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data['Feature'] = independent_vars_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(independent_vars_for_vif.values, i) for i in range(len(independent_vars_for_vif.columns))]

print(vif_data)

# Evaluate the model's performance on the test data
accuracy_stats_test = accuracy_score(test_df_stats['state_binary'], test_df_stats['predicted_class'])
conf_matrix_stats_test = confusion_matrix(test_df_stats['state_binary'], test_df_stats['predicted_class'])
class_report_stats_test = classification_report(test_df_stats['state_binary'], test_df_stats['predicted_class'])

print(f'Test Accuracy: {accuracy_stats_test}')
print('Test Confusion Matrix:')
print(conf_matrix_stats_test)
print('Test Classification Report:')
print(class_report_stats_test)

# Evaluate the model's performance on the training data
train_df_stats['predicted'] = stats_model_us.predict(train_df_stats)
train_df_stats['predicted_class'] = (train_df_stats['predicted'] > 0.5).astype(int)

accuracy_stats_train = accuracy_score(train_df_stats['state_binary'], train_df_stats['predicted_class'])
conf_matrix_stats_train = confusion_matrix(train_df_stats['state_binary'], train_df_stats['predicted_class'])
class_report_stats_train = classification_report(train_df_stats['state_binary'], train_df_stats['predicted_class'])

print(f'Training Accuracy: {accuracy_stats_train}')
print('Training Confusion Matrix:')
print(conf_matrix_stats_train)
print('Training Classification Report:')
print(class_report_stats_train)

#%% [markdown]
# We settled on looking at the US only subsetted data using the same variables as the regression model used for all countries. The accuracy of the model did not improve much when subsetting out all of the other countries. The VIF shows that there is not too much multicollinearity affecting the coefficients. 

# Training and the testing accuracy show that the model is not likely to be overfit and does a decent job at predicting campaign success or failure for US Kickstarter campaigns. 