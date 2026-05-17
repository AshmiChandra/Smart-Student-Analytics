import pandas as pd

df = pd.read_csv("dataset.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.columns)


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=df['study_hours'],y=df['ovall'])
plt.title("study hours vs overall score")
plt.xlabel('study hours')
plt.ylabel('overall score')
plt.savefig('visuals/study_vs_overall.png')
plt.show()

sns.scatterplot(x=df['attendace'],y=df['ovall'])
plt.savefig('visuals/att_vs_overall.png')
plt.show()

sns.histplot(x=df['attendace'])
plt.savefig('visuals/att_hist.png')
plt.show()

sns.barplot(x=df['school_type'],y=df['ovall'])
plt.savefig('visuals/schl_vs_overall.png')
plt.show()

sns.boxplot(x=df['ovall'])
plt.savefig('visuals/boxplot.png')
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.savefig('visuals/heatmap.png')
plt.show()