
# An Investigation of the External Factors Associated with the Medical Appointment Attendance in Brazil 

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

Patients can use medical appointments to reduce their wait time and improve the medical centres' service efficiency. It would be ideal that all the patients attend their doctor appointments on schedule. In reality, not all patients will go to their appointments because of various reasons. This project is to predict if a patient will show up for their appointments by investigating a dataset that has collected information from 100,000 medical appointments in Brazil. In the dataset, factors associated with appointments attendance are categorized into two groups: external and internal. External factors include elements related to the appointment system, such as appointment dates, SMS reminders, and locations of the hospitals. External elements probably vary for each appointment of the same patient. Internal factors include elements that are related to the patients themselves, such as their age, gender, and their long-term health conditions. Internal elements do not vary for each appointment of the same patient. The investigation only focuses on the external factors which are directly related to the medical appointment system, and the following questions will be addressed.

1. Would it be possible to improve the attendance rate by shortening the wait time between the scheduled day and appointment day?
2. Is SMS reminder useful on improving the attendance rate?
3. Which neighbourhood has the longest wait time?


```python
# import statements for all of the packages to be used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
```

<a id='wrangling'></a>
## Data Wrangling


### General Properties


```python
# Load and inspect data
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Observations:
1. Column titles are not in a consistent format
2. Typo found on "Handicap", and it should be "Handicap" (Not relevant to this project)



```python
# check data type and missing data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 14 columns):
    PatientId         110527 non-null float64
    AppointmentID     110527 non-null int64
    Gender            110527 non-null object
    ScheduledDay      110527 non-null object
    AppointmentDay    110527 non-null object
    Age               110527 non-null int64
    Neighbourhood     110527 non-null object
    Scholarship       110527 non-null int64
    Hipertension      110527 non-null int64
    Diabetes          110527 non-null int64
    Alcoholism        110527 non-null int64
    Handcap           110527 non-null int64
    SMS_received      110527 non-null int64
    No-show           110527 non-null object
    dtypes: float64(1), int64(8), object(5)
    memory usage: 11.8+ MB


Observations: 
1. There are 110527 entries and 14 columns, 
2. No null values
3. PatientId values are not integers (Not relevant to this project)
4. ScheduledDay and AppointmentDay are not in datetime format 


```python
# check unique entries of each column
df.nunique()
```




    PatientId          62299
    AppointmentID     110527
    Gender                 2
    ScheduledDay      103549
    AppointmentDay        27
    Age                  104
    Neighbourhood         81
    Scholarship            2
    Hipertension           2
    Diabetes               2
    Alcoholism             2
    Handcap                5
    SMS_received           2
    No-show                2
    dtype: int64



Observations: 
1. One patient can have several appointments
2. There are 104 unique values for Age (Not related to this project)
3. There are 5 unique values for Handicap (Not related to this project)


```python
# Check for duplicated entry
sum(df.duplicated())
```




    0



Observations:
1. No duplicated entry

### Data Cleaning

##### Step 1 Correct the Format


```python
# Drop unrelated columns for this projects
df_clean = df.drop(columns=['PatientId', 'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap'])

# Rename the columns so that all column heads are in a consistent format
df_clean.rename(columns = {'No-show':'No_show'},inplace = True)

# Change Day to datetime format
df_clean['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df_clean['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Check the output
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 6 columns):
    AppointmentID     110527 non-null int64
    ScheduledDay      110527 non-null datetime64[ns]
    AppointmentDay    110527 non-null datetime64[ns]
    Neighbourhood     110527 non-null object
    SMS_received      110527 non-null int64
    No_show           110527 non-null object
    dtypes: datetime64[ns](2), int64(2), object(2)
    memory usage: 5.1+ MB


##### Step 2 Modify Dataframe


```python
# Calculate the time interval between the scheduledDay and the appointmentDay, and name the differences as 'TimeGap'
df_clean['WaitDay'] = df_clean['AppointmentDay'] - df_clean['ScheduledDay']

# Convert the time difference into days
df_clean['WaitDay'] = df_clean.WaitDay.astype(pd.Timedelta).apply(lambda l: l.days)

# Check unique values for TimeGap
df_clean.WaitDay.unique()
```




    array([ -1,   1,   2,   0,   3,   8,  28,   9,  22,  10,  17,  16,  13,
            27,  23,  20,  14,  15,  21,  42,  29,  30,  41,  31,  55,  44,
            45,  38,  36,  37,  43,  49,  59,  51,  52,  64,  66,  90,  65,
            83,  77,  86, 114, 108,  62,  69,  71,  56,  57,  50,  58,  40,
            48,  72,  63,  19,  32,  33,   5,  34,  35,  11,  12,  39,  46,
             7,   4,   6,  24,  25,  47,  26,  18,  60,  54,  61, 175,  53,
            76,  68,  82,  75,  88,  80, 102,  78,  67,  74,  84, 111,  -2,
            79,  85,  97,  93, 141, 154, 161, 168, 103, 132, 124,  95,  87,
            89, 150, 125, 126, 110, 118,  73,  70,  81, 107, 109, 101, 121,
           100, 104,  91,  96,  92, 106,  94,  -7, 138, 131, 178, 116, 145, 122])



Observations:
1. Negative values existing in the array. These negative values can be filtered.


```python
# Filter the entries in the dataset with negative waitting values
df_clean = df_clean[df_clean['WaitDay'] >= 0]

# Check the dataset
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 71959 entries, 5 to 110526
    Data columns (total 7 columns):
    AppointmentID     71959 non-null int64
    ScheduledDay      71959 non-null datetime64[ns]
    AppointmentDay    71959 non-null datetime64[ns]
    Neighbourhood     71959 non-null object
    SMS_received      71959 non-null int64
    No_show           71959 non-null object
    WaitDay           71959 non-null int64
    dtypes: datetime64[ns](2), int64(3), object(2)
    memory usage: 4.4+ MB



```python
# Check the Waitting column
df_clean.WaitDay.unique()
```




    array([  1,   2,   0,   3,   8,  28,   9,  22,  10,  17,  16,  13,  27,
            23,  20,  14,  15,  21,  42,  29,  30,  41,  31,  55,  44,  45,
            38,  36,  37,  43,  49,  59,  51,  52,  64,  66,  90,  65,  83,
            77,  86, 114, 108,  62,  69,  71,  56,  57,  50,  58,  40,  48,
            72,  63,  19,  32,  33,   5,  34,  35,  11,  12,  39,  46,   7,
             4,   6,  24,  25,  47,  26,  18,  60,  54,  61, 175,  53,  76,
            68,  82,  75,  88,  80, 102,  78,  67,  74,  84, 111,  79,  85,
            97,  93, 141, 154, 161, 168, 103, 132, 124,  95,  87,  89, 150,
           125, 126, 110, 118,  73,  70,  81, 107, 109, 101, 121, 100, 104,
            91,  96,  92, 106,  94, 138, 131, 178, 116, 145, 122])



<a id='eda'></a>
## Exploratory Data Analysis


### Research Question 1: Would it be possible to improve the attendance rate by reducing the wait time between the scheduled day and appointment day?


```python
# Check statistical information of the Waitting column
df_clean.WaitDay.describe()
```




    count    71959.000000
    mean        14.642018
    std         16.494334
    min          0.000000
    25%          3.000000
    50%          8.000000
    75%         21.000000
    max        178.000000
    Name: WaitDay, dtype: float64



Oberservations:
1. 50% of patients scheduled their appointments in 8 days
2. 75% of patients scheduled their appointments in 21 days
3. The average wait time for a patient is 14 days
4. It would be beneficial to separate the wait time by weeks for analysis


```python
# Seperate time in different intervals
time_names = ['ThreeDays','OneWeek','TwoWeek','ThreeWeek','OneMonth','TwoMonth','OneQuarter','OverOneQuarter']
time_bin = [-1,3,7,14,21,30,60,90,200]

# Create new column to group the wait time in different time intervals
df_clean['WaitTime'] = pd.cut(df_clean['WaitDay'], bins = time_bin, labels = time_names)
df_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AppointmentID</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Neighbourhood</th>
      <th>SMS_received</th>
      <th>No_show</th>
      <th>WaitDay</th>
      <th>WaitTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5626772</td>
      <td>2016-04-27 08:36:51</td>
      <td>2016-04-29</td>
      <td>REPÚBLICA</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>ThreeDays</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5630279</td>
      <td>2016-04-27 15:05:12</td>
      <td>2016-04-29</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>Yes</td>
      <td>1</td>
      <td>ThreeDays</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5630575</td>
      <td>2016-04-27 15:39:58</td>
      <td>2016-04-29</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>Yes</td>
      <td>1</td>
      <td>ThreeDays</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5629123</td>
      <td>2016-04-27 12:48:25</td>
      <td>2016-04-29</td>
      <td>CONQUISTA</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>ThreeDays</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5630213</td>
      <td>2016-04-27 14:58:11</td>
      <td>2016-04-29</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>No</td>
      <td>1</td>
      <td>ThreeDays</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check if there is any null value
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 71959 entries, 5 to 110526
    Data columns (total 8 columns):
    AppointmentID     71959 non-null int64
    ScheduledDay      71959 non-null datetime64[ns]
    AppointmentDay    71959 non-null datetime64[ns]
    Neighbourhood     71959 non-null object
    SMS_received      71959 non-null int64
    No_show           71959 non-null object
    WaitDay           71959 non-null int64
    WaitTime          71959 non-null category
    dtypes: category(1), datetime64[ns](2), int64(3), object(2)
    memory usage: 4.5+ MB



```python
# Confirm all appointmentID are unbique
df_clean.AppointmentID.nunique()
```




    71959




```python
# To get an idea of the number of No-shows for each time interval
df_clean.groupby('WaitTime')['No_show'].value_counts().unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>No_show</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>WaitTime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ThreeDays</th>
      <td>15375</td>
      <td>4590</td>
    </tr>
    <tr>
      <th>OneWeek</th>
      <td>10700</td>
      <td>3852</td>
    </tr>
    <tr>
      <th>TwoWeek</th>
      <td>7700</td>
      <td>3496</td>
    </tr>
    <tr>
      <th>ThreeWeek</th>
      <td>5781</td>
      <td>2763</td>
    </tr>
    <tr>
      <th>OneMonth</th>
      <td>5380</td>
      <td>2616</td>
    </tr>
    <tr>
      <th>TwoMonth</th>
      <td>5106</td>
      <td>2640</td>
    </tr>
    <tr>
      <th>OneQuarter</th>
      <td>1278</td>
      <td>521</td>
    </tr>
    <tr>
      <th>OverOneQuarter</th>
      <td>117</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To seperate values counts for attendance and non-attendance
No_Show = df_clean.query('No_show == "Yes"').WaitTime.value_counts(sort=False)
Show = df_clean.query('No_show == "No"').WaitTime.value_counts(sort=False)

# Generate a dataframe for dataframe plotting
df_attendance = pd.DataFrame({'Show':Show,'No_Show':No_Show})

# Calculate the attendance proportion for each time group
Show_P = Show / (Show + No_Show)
Show_P
```




    ThreeDays         0.770098
    OneWeek           0.735294
    TwoWeek           0.687746
    ThreeWeek         0.676615
    OneMonth          0.672836
    TwoMonth          0.659179
    OneQuarter        0.710395
    OverOneQuarter    0.726708
    Name: WaitTime, dtype: float64




```python
# To create the plot
figure, ax1 = plt.subplots(figsize=(11,5))
df_attendance.plot(kind ='bar', rot=0, ax=ax1, color=tuple(["g", "r"]))
plt.ylabel('Attendance Count',fontsize = 12)
plt.xlabel('Wait Time',fontsize = 12)
plt.title('Attendance VS. Wait Time',fontsize = 12)

ax2 = ax1.twinx() 
ax2.set_ylim([0.5, 1])
sns.pointplot(x=Show_P.index, y=Show_P, color='b', ax=ax2)
plt.ylabel('Attendance Proportion',fontsize = 12)
```




    Text(0,0.5,'Attendance Proportion')




<img src="images/output_27_1.png?raw=true"/>


### Research Question 2: Is SMS reminder effecitve on improving the attendance rate?


```python
# To account for all the no shows for both SMS received (1) and SMS not received (0) cases
SMS1 = df_clean.query('No_show == "Yes"').SMS_received.value_counts(sort=False)
SMS1
```




    0    10738
    1     9784
    Name: SMS_received, dtype: int64




```python
# To account for all the shows for both SMS received (1) and SMS not received (0) cases
SMS2 = df_clean.query('No_show == "No"').SMS_received.value_counts(sort=False)
SMS2
```




    0    25739
    1    25698
    Name: SMS_received, dtype: int64




```python
# Calculate the change in attendance rate from SMS not received (0) to SMS received (1)
SMS = SMS2 / (SMS1 + SMS2)
SMS
```




    0    0.705623
    1    0.724255
    Name: SMS_received, dtype: float64




```python
# To creat the plot
df_SMS = pd.DataFrame({'Show':SMS2,'No_Show':SMS1})

figure, ax1 = plt.subplots(figsize=(7,4))
df_SMS.plot(kind ='bar', rot=0, ax=ax1, color=tuple(["g", "r"]))
plt.ylabel('Attendance Count',fontsize = 12)
plt.xlabel('SMS not received (0), SMS received (1)',fontsize = 12)
plt.title('SMS Influence on Attendance',fontsize = 12)

ax2 = ax1.twinx() 
ax2.set_ylim([0.6, 0.75])
sns.pointplot(x=SMS.index, y=SMS, color='b', ax=ax2)
plt.ylabel('Attendance Proportion',fontsize = 12)
```




    Text(0,0.5,'Attendance Proportion')




![png](output_32_1.png)


### Research Question 3: Which neighbourhood has the longest wait time?


```python
# Calculate the average wait days for each neighbourhood
kk = df_clean.groupby('Neighbourhood').mean().WaitDay

# Sort the neighbourhood with wait days
ag = kk.sort_values(ascending=False)

# Select the neighbourhood which has more than 20 average wait days
mm = ag.loc[lambda x : x>=20]
mm
```




    Neighbourhood
    ILHAS OCEÂNICAS DE TRINDADE    28.000000
    SANTA CECÍLIA                  27.174785
    JARDIM CAMBURI                 26.773067
    FONTE GRANDE                   24.980810
    AEROPORTO                      22.600000
    MARUÍPE                        21.994849
    JUCUTUQUARA                    20.650000
    JOANA D´ARC                    20.647887
    Name: WaitDay, dtype: float64




```python
# To create the plot 
ind = np.arange(len(mm))
width = 0.6
labels = mm.index.str.title()
plt.figure(figsize=(18, 5))
plt.bar(ind, mm, width, color='y', alpha=0.8, tick_label=labels)
plt.ylabel('Average wait days',fontsize = 12)
plt.xlabel('Neighbourhood',fontsize = 12)
plt.title('Neighbourhoods with the most average wait days',fontsize = 12)
```




    Text(0.5,1,'Neighbourhoods with the most average wait days')




![png](output_35_1.png)


<a id='conclusions'></a>
## Conclusions

The first figure shows that there is a negative relationship between the attendance rate and the wait time. Long wait times can cause a low attendance rate. Interestingly, the attendance rate is going up when the wait time is above two months. The finding may indicate that patients are more likely to attend to their appointments, which are booked for recent days or two months after. However, the evidence supporting this claim is not reliable because there is not enough data. 

The second figure shows that SMS reminders can slightly improve the attendance rate. The first group of columns on the graph shows the attendance of the patients who did not receive the SMS reminder. The second group of columns shows the attendance of the patients who received the SMS reminder. The proportions of attendance and non-attendance are similar for each case. 

The third graph selected 8 neighbourhoods that have more than 20 wait days on average. We would suggest that more medical supports should be given in these neighbourhoods. 


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```




    0


