---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---


1.  Write a program to print out the grade of a person according to the
    following criteria:

-   less than 50: C
-   between 50 and 80: B
-   between 80 and 90: A
-   above 90: A+

Hint: do you have enough information to write a program?


``` python
# Input: Score of the person
score = float(input("Enter the score: "))

# Determine the grade
if score < 50:
    grade = "C"
elif 50 <= score < 80:
    grade = "B"
elif 80 <= score <= 90:
    grade = "A"
else:  # score > 90
    grade = "A+"

# Output the grade
print(f"The grade is: {grade}")
```

    Enter the score: 100
    The grade is: A+

1.  What\'s the output here?

This code will throw a NameError because the variable hour in
print(number_of_minutes(hour)) is not defined. The correct variable name
is hours.

``` python
hours = 3

def number_of_minutes(hours):
  MINUTES_PER_HOUR = 60
  return hours * MINUTES_PER_HOUR

# Changed 'hour' to 'hours' in the function call
print(number_of_minutes(hours))
```

180

1.  Write a function to return the number of hours for a given number of
    days

``` python
def number_of_hours(days):
    HOURS_PER_DAY = 24
    return days * HOURS_PER_DAY

days = 7
print(f"{days} days equals {number_of_hours(days)} hours.")
```

    7 days equals 168 hours.

1.  You are given a series of monthly sales revenue (in thousand USD),
    write a series of functions to compute the following descriptive
    statistics:

-   Mean
-   Median
-   Mode
-   Variance
-   Standard deviation
-   Count
-   Range

``` python
import statistics

sales_revenue = [
    150, 200, 210, 190, 170, 220, 250, 230, 210, 180, 170, 160,
    300, 190, 175, 160, 140, 240, 260, 300, 310, 330, 290, 210
]

def compute_statistics(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    mode = statistics.mode(data)
    variance = statistics.variance(data)
    std_dev = statistics.stdev(data)
    count = len(data)
    range_value = max(data) - min(data)

    return {
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Count": count,
        "Range": range_value,
    }

stats = compute_statistics(sales_revenue)
for key, value in stats.items():
    print(f"{key}: {value}")
```

    Mean: 218.54166666666666
    Median: 210.0
    Mode: 210
    Variance: 3064.085144927536
    Standard Deviation: 55.35417910986971
    Count: 24
    Range: 190

1.  You have a dataset of product categories sold in a department store.
    Calculate descriptive statistics describing the data:

-   Total number of products sold
-   Unique product categories
-   Frequencies of each product category
-   Most frequently sold product category

``` python
from collections import Counter

product_categories = [
    "Electronics", "Clothing", "Electronics", "Home & Garden", "Clothing", "Beauty",
    "Clothing", "Electronics", "Toys", "Home & Garden", "Clothing", "Clothing",
    "Beauty", "Electronics", "Toys", "Clothing", "Electronics", "Home & Garden",
    "Electronics", "Clothing", "Beauty", "Toys"
]

def analyze_product_categories(categories):
    total_products = len(categories)
    unique_categories = set(categories)
    category_frequencies = Counter(categories)
    most_common_category = category_frequencies.most_common(1)[0]

    return {
        "Total Products": total_products,
        "Unique Categories": unique_categories,
        "Category Frequencies": category_frequencies,
        "Most Common Category": most_common_category,
    }

category_stats = analyze_product_categories(product_categories)
for key, value in category_stats.items():
    print(f"{key}: {value}")
```

    Total Products: 22
    Unique Categories: {'Clothing', 'Home & Garden', 'Beauty', 'Electronics', 'Toys'}
    Category Frequencies: Counter({'Clothing': 7, 'Electronics': 6, 'Home & Garden': 3, 'Beauty': 3, 'Toys': 3})
    Most Common Category: ('Clothing', 7)

------------------------------------------------------------------------

``` python
from google.colab import drive
drive.mount("/content/drive")
```

    Mounted at /content/drive

``` python
# Import relevant libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

``` python
# Change the path to point to your data location
filepath = "/content/drive/MyDrive/DataCamp/healthcare-dataset-stroke-data.csv"

# Read the CSV file into a Pandas DataFrame
stroke_data = pd.read_csv(filepath)

# Print the DataFrame
print(stroke_data)
```

             id  gender   age  hypertension  heart_disease ever_married  \
    0      9046    Male  67.0             0              1          Yes   
    1     51676  Female  61.0             0              0          Yes   
    2     31112    Male  80.0             0              1          Yes   
    3     60182  Female  49.0             0              0          Yes   
    4      1665  Female  79.0             1              0          Yes   
    ...     ...     ...   ...           ...            ...          ...   
    5105  18234  Female  80.0             1              0          Yes   
    5106  44873  Female  81.0             0              0          Yes   
    5107  19723  Female  35.0             0              0          Yes   
    5108  37544    Male  51.0             0              0          Yes   
    5109  44679  Female  44.0             0              0          Yes   

              work_type Residence_type  avg_glucose_level   bmi   smoking_status  \
    0           Private          Urban             228.69  36.6  formerly smoked   
    1     Self-employed          Rural             202.21   NaN     never smoked   
    2           Private          Rural             105.92  32.5     never smoked   
    3           Private          Urban             171.23  34.4           smokes   
    4     Self-employed          Rural             174.12  24.0     never smoked   
    ...             ...            ...                ...   ...              ...   
    5105        Private          Urban              83.75   NaN     never smoked   
    5106  Self-employed          Urban             125.20  40.0     never smoked   
    5107  Self-employed          Rural              82.99  30.6     never smoked   
    5108        Private          Rural             166.29  25.6  formerly smoked   
    5109       Govt_job          Urban              85.28  26.2          Unknown   

          stroke  
    0          1  
    1          1  
    2          1  
    3          1  
    4          1  
    ...      ...  
    5105       0  
    5106       0  
    5107       0  
    5108       0  
    5109       0  

    [5110 rows x 12 columns]

``` python
stroke_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5110 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5110 non-null   int64  
     1   gender             5110 non-null   object 
     2   age                5110 non-null   float64
     3   hypertension       5110 non-null   int64  
     4   heart_disease      5110 non-null   int64  
     5   ever_married       5110 non-null   object 
     6   work_type          5110 non-null   object 
     7   Residence_type     5110 non-null   object 
     8   avg_glucose_level  5110 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     5110 non-null   object 
     11  stroke             5110 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 479.2+ KB

Each row in the data provides relavant information about the patient.

1.  id: unique identifier
2.  gender: \"Male\", \"Female\" or \"Other\"
3.  age: age of the patient
4.  hypertension: 0 if the patient doesn\'t have hypertension, 1 if the
    patient has hypertension
5.  heart_disease: 0 if the patient doesn\'t have any heart diseases, 1
    if the patient has a heart disease
6.  ever_married: \"No\" or \"Yes\"
7.  work_type: \"children\", \"Govt_jov\", \"Never_worked\", \"Private\"
    or \"Self-employed\"
8.  Residence_type: \"Rural\" or \"Urban\"
9.  avg_glucose_level: average glucose level in blood
10. bmi: body mass index
11. smoking_status: \"formerly smoked\", \"never smoked\", \"smokes\" or
    \"Unknown\"\*
12. stroke: 1 if the patient had a stroke or 0 if not

-   Note: \"Unknown\" in smoking_status means that the information is
    unavailable for this patient

**Exercises:**

*Which columns are numerical and which ones are categorical?*

Numerical columns:

-   age: Continuous numerical data.
-   avg_glucose_level: Continuous numerical data.
-   bmi: Continuous numerical data (with some missing values).
-   hypertension: Binary numerical data (0 or 1).
-   heart_disease: Binary numerical data (0 or 1).
-   stroke: Binary numerical data (0 or 1).

Categorical columns:

-   id: Identifier (although numerical, it's categorical since it
    uniquely
-   identifies each patient).
-   gender: Categorical (Male, Female, Other).
-   ever_married: Categorical (Yes, No).
-   work_type: Categorical (children, Govt_job, Never_worked, Private,
    Self-employed).
-   Residence_type: Categorical (Rural, Urban).
-   smoking_status: Categorical (formerly smoked, never smoked, smokes,
    Unknown).

*Why is it important to analyze numerical and non-numerical columns
separatey?*

-   Different statistical methods: Numerical and categorical data
    require different statistical techniques for analysis.
-   Feature engineering: Categorical data often needs to be encoded
    before being used in machine learning models.
-   Interpretation: The insights derived from numerical and categorical
    variables can be quite different.

**Formulate Data Questions** (Hypothesis)

1.  *What valuable insights can we discover from the data?*

We can identify potential relationships or trends in the data, such as:
The proportion of patients experiencing strokes based on factors like
age, gender, or health conditions (e.g., hypertension, heart disease).
The impact of glucose levels or BMI on stroke risk. Differences between
groups (e.g., urban versus rural residents).

1.  *Which questions will be most relevant or engaging for
    stakeholders?*

-   Which factors pose the highest risk for strokes?
-   How can we identify patient groups that should be prioritized for
    prevention?
-   Should targeted healthcare programs be implemented for specific
    groups (based on age, lifestyle, or location)?

**Exploratory Data Analysis**

Before we start, let\'s revise some concepts related to distributions:

1.  *What is a distribution?*

A distribution shows how data values are spread across a dataset. For
example, a normal distribution will have most values clustered around
the mean and fewer values at the extremes.

1.  *What is the range of a distribution?*

The range is the difference between the largest and smallest values in a
dataset. For example, if age values range from 20 to 80, the range is
60.

1.  *What are quartiles and the interquartile range (IQR)?*

Quartiles: These divide the dataset into four equal parts. Interquartile
Range (IQR): This is the difference between the first quartile (Q1) and
the third quartile (Q3), used to detect outliers.

1.  *What are mean, mode, and median values?*

-   Mean: The sum of all values divided by the total number of values.
-   Median: The middle value when the data is sorted in order.
-   Mode: The value that appears most frequently in the dataset

**Check for missing values** Why? Missing values is number one problem
with a dataset because it affects almost every aspect of your downstream
analysis. For example:

1.  *How do missing values affect descriptive statistics?*

When values are missing, statistics like the mean or median may not
accurately represent the dataset. For example, if many patients\' BMI
values are missing, the average BMI may be misleading, resulting in
incorrect risk assessments for the group.

1.  *How do missing values impact predictive models?*

Missing data can reduce the performance of predictive models. For
example, machine learning models may struggle to process missing values
if proper handling (e.g., imputation or removal) isn't done. Model
quality can be compromised, leading to inaccurate predictions.

**Dealling with Categorical and Numerical Variables**

col_names() - This a function created to get the columns names that has
categorical and numerical data separately

``` python
def col_names(df):
    # Get categorical Variables
    cat_cols = [col for col in df.columns if df[col].dtypes not in ["int64", "float64"]]
    # Get numerical Variables
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    # Get Numerical but Categorical Variables
    num_but_cat = [col for col in num_cols if df[col].nunique() < 10]
    # Adding num_but_cat to cat_cols
    cat_cols = num_but_cat + cat_cols
    # num_but_cat removing from num_cols
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Numerical Cols: {num_cols} \nCategorical Cols: {cat_cols} \nNumerical but Categorical: {num_but_cat}")
    return num_cols, cat_cols
```

``` python
numerical_cols, categorical_cols = col_names(stroke_data)
```

    Numerical Cols: ['id', 'age', 'avg_glucose_level', 'bmi'] 
    Categorical Cols: ['hypertension', 'heart_disease', 'stroke', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] 
    Numerical but Categorical: ['hypertension', 'heart_disease', 'stroke']

``` python
# Visuealize each attribute
sns.histplot(x=stroke_data["age"], data=stroke_data, color="blue")
```
![](vertopal_4db5c5a4c08e445ebcc155088842d070/ca64389078bca9516952181a2f2386deebfa28be.png)
**Exercises:**

-   Can you plot the histograms of other numerical columns?
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":928}" id="7zVsMKELKMUk" outputId="dd19cb1b-0b58-4876-bfac-6aace8ded05e"}
``` python
# Plot histogram for avg_glucose_level
sns.histplot(x=stroke_data["avg_glucose_level"], color="green", kde=True)
plt.title("Distribution of Avg Glucose Level")
plt.show()

# Plot histogram for bmi
sns.histplot(x=stroke_data["bmi"], color="orange", kde=True)
plt.title("Distribution of BMI")
plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/6838ba26c3ee5e89b7d76dfb94f3b8a2c927af33.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/3ebeb9a00ff6de45ee807b79b76731c1bab9ed88.png)
:::
:::

::: {.cell .markdown id="35OWmFRi6RR-"}
**Exercises:**

-   Can you show the plots of \"age\", \"avg_glucose_level\", \"bmi\"
    side by side, i.e. 3 plots on the same row?
:::

::: {.cell .code execution_count="29" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":508}" id="ThW12bo5Kl8p" outputId="27ad7145-f99a-4be9-da9b-744c752f0872"}
``` python
# Create subplots for age, avg_glucose_level, and bmi
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for age
sns.histplot(x=stroke_data["age"], ax=axes[0], color="blue")
axes[0].set_title("Age Distribution")

# Plot for avg_glucose_level
sns.histplot(x=stroke_data["avg_glucose_level"], ax=axes[1], color="green")
axes[1].set_title("Avg Glucose Level")

# Plot for bmi
sns.histplot(x=stroke_data["bmi"], ax=axes[2], color="orange")
axes[2].set_title("BMI Distribution")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/107a77e993e7f5fee9b0e00f053448d5c8f389d0.png)
:::
:::

::: {.cell .markdown id="GV44Z5ZKKb0v"}
**Exercises:**

-   Can you show customize the axes of the plots to show \"Age\" ,
    \"Average Gluscose Level\", \"BMI\" on the x-axes?
-   References:
    -   <https://seaborn.pydata.org/generated/seaborn.histplot.html>
    -   <https://www.w3schools.com/python/matplotlib_histograms.asp>
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="-GR-W_gaKuXs" outputId="4782751e-7609-444e-86a0-7d29b933bd30"}
``` python
# Customized histogram plots
sns.histplot(x=stroke_data["age"], color="blue")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()

sns.histplot(x=stroke_data["avg_glucose_level"], color="green")
plt.xlabel("Average Glucose Level")
plt.ylabel("Count")
plt.title("Glucose Level Distribution")
plt.show()

sns.histplot(x=stroke_data["bmi"], color="orange")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.title("BMI Distribution")
plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/77dad8c862892c90e9154511444a5c5ad8550139.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/acfc1fe1ac38076d42a24c000347e943bcdf89ec.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/6be3250b81efd13f401f0870db8a675806b5e8e2.png)
:::
:::

::: {.cell .markdown id="nc36gQ8X6YTq"}
**Check for Outliers** Outliers are another \"defect\" in the data that
need \"correction\":

1.  *Outliers can significantly affect some statistics of a
    distribution, which can affect both your interpretations + how you
    fix missing data. Can you figure out why is that so?*

Impact on Statistics:

-   Outliers can skew summary statistics like the mean, standard
    deviation, and variance, which are sensitive to extreme values.
    While the median and mode are less sensitive to outliers, they can
    still shift if the dataset is small or has concentrated values near
    the outlier.
-   Impact on Missing Data Fixing: When handling missing values, methods
    like mean imputation or interpolation use statistical properties of
    the data. If the mean is skewed by outliers, missing values may be
    replaced with unrealistic or extreme values, distorting the overall
    data distribution.

1.  *Outliers can significantly affect predictive models. Can you figure
    out why it is so?*

Predictive models, especially those that rely on minimizing errors
(e.g., regression), can be disproportionately influenced by outliers.
For instance: Linear regression tries to minimize the sum of squared
errors, and outliers can have a significant impact on the slope or
intercept. Decision trees might split based on extreme values, leading
to overfitting.

Models may try to fit the outliers instead of learning general patterns,
reducing their ability to generalize to unseen data. Underperformance:
The model might give too much weight to outliers, leading to poor
predictions for normal data points.

Outliers should be carefully handled during exploratory data analysis
(EDA):

-   Use visualizations like boxplots, histograms, or scatter plots to
    identify outliers.
-   Consider methods to handle outliers, such as:
-   Removing extreme values if they're errors or irrelevant.
-   Using robust statistical measures (median, interquartile range)
    instead of the mean.
-   Applying transformations (e.g., log or square root) to reduce their
    impact.
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":718}" id="ulWo4euq6jEL" outputId="5e282b85-38a3-4895-f82b-2d8ec9232db7"}
``` python
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
# Checking for outliers of "BMI"
columns = ["age", "avg_glucose_level", "bmi" ]
sns.boxplot(x=stroke_data["bmi"], data=stroke_data, color="indianred")
```

::: {.output .execute_result execution_count="9"}
    <Axes: xlabel='bmi'>
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/fb45c995d7a1f80870342bef422b0032146d6a84.png)
:::
:::

::: {.cell .markdown id="ZFvoR59PTfuH"}
What do the box, \"whiskers\", and dots indicate?

**Exercises:**

-   Can you plot the histograms of other numerical columns?
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="1nU8xKoDTaFt" outputId="8cd8764f-476f-46a7-acfc-155905ed1fad"}
``` python
# Plot histograms for numerical columns
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
for col in numerical_cols:
    sns.histplot(stroke_data[col], kde=True, color="blue")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/fde00672780273ab7bbe216f470c883022d4a318.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/d3a7e1d17ee364641a224b48da93cd933c9f3e8f.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/5c969ccb9ba3c24f98ef4891a450603e97070da9.png)
:::
:::

::: {.cell .markdown id="sHnnvNa1OOCY"}
**Exercises:**

-   Can you show the box plots of \"age\", \"avg_glucose_level\",
    \"bmi\" side by side, i.e. 3 plots on the same row, each plot is
    vertically-oriented instead of the default horizontal orientation
-   References:
    -   <https://seaborn.pydata.org/generated/seaborn.boxplot.html>
    -   <https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/>
:::

::: {.cell .code execution_count="36" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":452}" id="KllI9PocTsSz" outputId="fc769e1c-c8bd-4440-fa59-66ab69afb20b"}
``` python
# Create vertically-oriented boxplots
sns.boxplot(data=stroke_data[numerical_cols], orient="v", palette="Set2")
plt.title("Boxplots of Numerical Columns")
plt.ylabel("Values")
plt.xticks(ticks=range(len(numerical_cols)), labels=numerical_cols)
plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/2c73e4d801e1423d8effd43ca569369974a9158f.png)
:::
:::

::: {.cell .markdown id="jZ5HHqgBOPwZ"}
#### \[Extra\] Checking for Outliers in Numerical Attributes

IQR (Interqartile range) is being used to check for outliers. There are
other methods to do this, this snippet is mainly to practice the concept
of functions in Python.

check_outliers() - This function will return the columns with outliers
:::

::: {.cell .code execution_count="20" id="S4N__CdQOVtE"}
``` python
def check_outliers(df, numerical_cols, iqr=1.5, low_threshold=0.1, up_threshold=0.9):
    outlier_cols = []
    for col in numerical_cols:
        q1 = df[col].quantile(low_threshold)
        q3 = df[col].quantile(up_threshold)
        interquantile = q3 - q1
        up_limit = q3 + iqr * interquantile
        low_limit = q1 - iqr * interquantile
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            outlier_cols.append(col)
    if not outlier_cols:
        print("There is no outliers")
    return outlier_cols
```
:::

::: {.cell .code execution_count="15" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="YfOG66GWObKW" outputId="16e5fd2b-0dfd-4153-a958-5dbb4088a6bd"}
``` python
outlier_cols = check_outliers(stroke_data, ['age'])
```

::: {.output .stream .stdout}
    There is no outliers
:::
:::

::: {.cell .markdown id="kzoMGCUkOfqh"}
**Exercises**: Can you update the function call above to check for
outliers for all numerical columns?
:::

::: {.cell .code execution_count="42" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Od2F4OerlEJn" outputId="f76cade7-22d1-48b1-9abc-3873695cfc31"}
``` python
# Automatically identify numerical columns
numerical_cols = stroke_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Check for outliers across all numerical columns
outlier_cols = check_outliers(stroke_data, numerical_cols)

# Print results
if outlier_cols:
    print(f"Columns with outliers: {outlier_cols}")
else:
    print("No outliers detected in numerical columns.")
```

::: {.output .stream .stdout}
    Columns with outliers: ['hypertension', 'heart_disease', 'bmi', 'stroke']
:::
:::

::: {.cell .code execution_count="19" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":494}" id="pfn7p15yOv-2" outputId="dfc2a9b2-0ee0-4f5b-eb8c-b6af2b0f5be6"}
``` python
# Plot as percentage
plt.pie(stroke_data["stroke"].value_counts(), labels=[1,0], colors=["indianred", "mistyrose"])
```

::: {.output .execute_result execution_count="19"}
    ([<matplotlib.patches.Wedge at 0x7818790c5db0>,
      <matplotlib.patches.Wedge at 0x7818790c5cf0>],
     [Text(-1.0871361453364168, 0.16773491438301516, '1'),
      Text(1.087136143373357, -0.1677349271061446, '0')])
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/30b9d4e28eae1fecfde6730c45dd5b3400c26e40.png)
:::
:::

::: {.cell .markdown id="h4lyCMClO-Oy"}
Ratio of 1s is 95.1%
:::

::: {.cell .markdown id="VArPuj6lmdLX"}
**Exercises**

-   Can you update the labels to \"Stroke\" and \"No stroke\" instead of
    \"0\" and \"1\"?
:::

::: {.cell .code execution_count="43" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":428}" id="L4TcWzP5mZlp" outputId="94fd4a1f-3e0b-4d30-8fb8-5f319d57e859"}
``` python
plt.pie(stroke_data["stroke"].value_counts(),
        labels=["No Stroke", "Stroke"],
        colors=["mistyrose", "indianred"],
        autopct='%1.1f%%')  # Adding percentages for better visualization
plt.title("Stroke Distribution")
plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/03f15942f1d9d8037d1c04cc280bf33676f5fe38.png)
:::
:::

::: {.cell .markdown id="6STR6FsHO2U2"}
**Exercises** *Can you comment on the distribution of \"Stroke\" vs \"No
stroke\" labels, how would this affect your predictive model?* The
\"Stroke\" variable shows a highly imbalanced distribution, with 95.1%
of cases labeled as \"No Stroke\" and only 4.9% as \"Stroke.\" This
reflects the rarity of strokes but also poses challenges for predictive
modeling due to the underrepresentation of positive cases.

1.  Effect on the Predictive Model

-   Challenges of Imbalanced Data:
-   Model Bias: Machine learning models often prioritize the majority
    class (\"No Stroke\"), leading to poor detection of strokes.
-   Misleading Metrics: Accuracy could appear high (\~95%) even if the
    model predicts \"No Stroke\" for all cases, ignoring the critical
    minority class.

1.  Insights and Limitations:

-   The imbalance might reflect the true prevalence of strokes in the
    population but could also result from sampling bias or incomplete
    data collection. For instance, underrepresentation of high-risk
    groups (e.g., elderly or individuals with certain pre-existing
    conditions) could skew the dataset.
-   A deeper analysis by demographic or lifestyle factors (e.g., age,
    gender, alcohol consumption) is needed to verify if the imbalance is
    natural or a data issue.

1.  Mitigation Strategies:

-   To improve prediction for the minority class, we could use
    techniques Data Resampling like SMOTE to oversample \"Stroke\" cases
    or undersample \"No Stroke\" cases to balance the dataset.
-   Class Weights: Algorithms like Logistic Regression or Random Forest
    can assign higher importance to \"Stroke\" cases.
-   Evaluation Metrics: Metrics like F1-score, precision, recall, or
    AUC-ROC are better suited for imbalanced datasets than accuracy.
    Hypothesis and Exploration:

Analyzing stroke rates across demographics might reveal insights. For
example, if males have a higher stroke rate, it could suggest lifestyle
factors like alcohol consumption or unhealthy eating. However, this
could also reflect data collection bias (e.g., males being
overrepresented or females being underdiagnosed).
:::

::: {.cell .markdown id="Y1dxjMglPEAW"}
**Correlation between numerical columns**
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":458}" id="cCPD1YZVPF0K" outputId="d7a25b76-b3b6-4cc5-edc1-d91bbf97696e"}
``` python
sns.heatmap(stroke_data[numerical_cols].corr(), annot=True, linewidths=0.5,)
```

::: {.output .execute_result execution_count="33"}
    <Axes: >
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/6cf7cc03867a233a6bab1310a40d6aa004f2eb8b.png)
:::
:::

::: {.cell .markdown id="HAW4ymYNQ21r"}
**Exercises**

-   What is correlation coefficient? How is it computed and why it is
    important?
-   Why we only compute corellation coefficients between numerical
    variables?
-   Why is it important (or may not be important) to consider
    correlation between varibles? See here
    <https://www.widsworldwide.org/get-inspired/blog/a-data-scientists-deep-dive-into-the-wids-datathon/>
:::

::: {.cell .markdown id="L4FrJsuEQ7hJ"}
**Target Variable vs Numerical Attributes**
:::

::: {.cell .code execution_count="34" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="_G4ULhgoQ9DD" outputId="bb78749d-24aa-4d4c-da6b-bdb8c0180624"}
``` python
for col in numerical_cols:
    print(stroke_data.groupby("stroke").agg({col:"mean"}), end="\n\n")
```

::: {.output .stream .stdout}
                      id
    stroke              
    0       36487.236371
    1       37115.068273

                  age
    stroke           
    0       41.971545
    1       67.728193

            avg_glucose_level
    stroke                   
    0              104.795513
    1              132.544739

                  bmi
    stroke           
    0       28.823064
    1       30.471292
:::
:::

::: {.cell .markdown id="Q-5QnNyPRBWD"}
**Exercises**

**Can you visualize these results?**
:::

::: {.cell .code execution_count="44" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="AlUA5jH_pY1a" outputId="d06dbf35-3e39-4060-c40f-62c4e9dba893"}
``` python
import seaborn as sns
import matplotlib.pyplot as plt

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="stroke", y=col, data=stroke_data, palette="pastel")
    plt.title(f"Distribution of {col} by Stroke")
    plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/f657b8a2965ea345b16b57b927bf4a62ffa0a905.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/14c742c9e0b761fc242a93e6d1385c76a606c467.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/19555ee16b8028c9ebb9f4c22dd5073d76463d3c.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/a4c7aeb1ca5dfd3a03a8ede81c36fc6021ad3807.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/1dd9f1ae6aee5e10f03166f24e47099e0bfa392e.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/27962b667bdd1497d60182183235eec679433f36.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/4b071ab7eac9032f1118e02e3e99dc980c3733b5.png)
:::
:::

::: {.cell .markdown id="oWnGlNd9pfWT"}
**Exercises** **What interpretation can you draw from these results?**

From the aggregated means:

-   Age: Stroke patients have a higher mean age (67.73) compared to
    non-stroke individuals (41.97), indicating older adults are at
    higher risk.
-   Average Glucose Level: Stroke cases have higher mean glucose levels
    (132.54 vs. 104.80), suggesting a possible link between
    hyperglycemia and stroke.
-   BMI: Stroke patients also have a higher mean BMI (30.47 vs. 28.82),
    reflecting a potential relationship between obesity and stroke.

**Interpretation:**

These numerical trends align with known risk factors for stroke (e.g.,
age, high glucose, high BMI). However, causal relationships cannot be
confirmed due to potential confounding factors or biases in data
collection.
:::

::: {.cell .markdown id="M4bvNNmFRwFR"}
**Target Variable vs Categorical Attributes**
:::

::: {.cell .markdown id="fbeS_FH8R0fs"}
**Exercises**

**Can you repeat the computation above for categorical columns?**
:::

::: {.cell .code execution_count="47" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="eh67bXBAqEUu" outputId="b157c1d1-bba2-42ea-e5c2-4d09f5ecf53a"}
``` python
categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married']
for col in categorical_cols:
    print(stroke_data.groupby("stroke")[col].value_counts(normalize=True), "\n")
```

::: {.output .stream .stdout}
    stroke  gender
    0       Female    0.586916
            Male      0.412878
            Other     0.000206
    1       Female    0.566265
            Male      0.433735
    Name: proportion, dtype: float64 

    stroke  hypertension
    0       0               0.911129
            1               0.088871
    1       0               0.734940
            1               0.265060
    Name: proportion, dtype: float64 

    stroke  heart_disease
    0       0                0.952890
            1                0.047110
    1       0                0.811245
            1                0.188755
    Name: proportion, dtype: float64 

    stroke  ever_married
    0       Yes             0.644518
            No              0.355482
    1       Yes             0.883534
            No              0.116466
    Name: proportion, dtype: float64 
:::
:::

::: {.cell .markdown id="DwIpM2TSqIvk"}
**Exercises**

**Can you also plot the results?**
:::

::: {.cell .code execution_count="46" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="dd9A9rJUqbBE" outputId="41657159-f412-4c29-a29c-4a5acd67483a"}
``` python
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="stroke", data=stroke_data, palette="pastel")
    plt.title(f"{col} Distribution by Stroke")
    plt.show()
```

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/eb18117e708b5b1fe063775c9aa1ff466cafe952.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/2d7df0a4207b39f2e26c31570e668287f72836df.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/950ba9da86dc5c982c709fc89ffd6366255e7874.png)
:::

::: {.output .display_data}
![](vertopal_4db5c5a4c08e445ebcc155088842d070/bc37e517fdd2fe7ae2633b630409618d8661fd9c.png)
:::
:::

::: {.cell .markdown id="uqsrXulXqbV7"}
**Exercises**

**What insights can you draw from the results?**

-   Gender: If men have a higher stroke rate, it could reflect riskier
    behaviors (e.g., smoking, alcohol use).
-   Hypertension & Heart Disease: Elevated rates in stroke cases could
    signal these as major contributing factors.
-   Marriage Status: If stroke is more prevalent in married individuals,
    it might indicate older age rather than marital status directly
    causing strokes.
:::
