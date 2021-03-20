#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#1,Load data
df=pd.read_csv("C:/Users/gengl/Desktop/Artificial Intel and Mach Learn/project1/city_temperature.csv")


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.corr()


# In[6]:


#dropping the duplicate rows
df=df.drop_duplicates()
df.shape


# In[7]:


df.count()


# In[8]:


#dealing with the missing or null values
#check missing values(Nan) in every column
for col in df.columns:
    print("The "+ col +" contains Nan" + ":" + str((df[col].isna().any())))


# In[9]:


#check missing values(Nan) in every column
for col in df.columns:
    print("The "+ col +" contains 0" + ":" + str((df[col]==0).any()))


# In[10]:


#when i explore data, i fund that there is 0 in Day, is Outliers
df = df[df.Day !=0]
df.head()


# In[11]:


#when i explore data, i fund that there is 200 and 201 in year, is Outliers
df = df[(df.Year != 200) & (df.Year != 201)]
df.head()


# In[12]:


#when i explore data, i fund that there is -99 in AvgTemperature, is Outliers
df = df[df.AvgTemperature != -99]
df.head()


# In[16]:


#no outliers now, data is ready
df.info()


# In[13]:


#Find average Temperture in every region
#Data visualization 1
Average_Temperture_in_every_region = df.groupby("Region")["AvgTemperature"].mean().sort_values()[-1::-1]
Average_Temperture_in_every_region = Average_Temperture_in_every_region.rename({"South/Central America & Carribean":"South America","Australia/South Pacific":"Australia"})
Average_Temperture_in_every_region


# In[18]:


import matplotlib.pyplot as plt
plt.figure(figsize = (15,8))
plt.bar(Average_Temperture_in_every_region.index,Average_Temperture_in_every_region.values)
plt.xticks(rotation = 10,size = 15)
plt.yticks(size = 15)
plt.ylabel("Average_Temperture",size = 15)
plt.title("Average Temperture in every region",size = 20)
plt.show()


# In[19]:


# find growth of the average Temperture in every region over time
#Data visualization 2
#1 change the index to date
datetime_series = pd.to_datetime(df[['Year','Month', 'Day']])
df['date'] = datetime_series
df = df.set_index('date')
df = df.drop(["Month","Day","Year"],axis = 1)
df.head()


# In[20]:


region_year = ['Region', pd.Grouper(freq='Y')]
df_region = df.groupby(region_year).mean()
df_region.head()


# In[22]:


plt.figure(figsize = (20,10))
for region in df["Region"].unique():

    plt.plot((df_region.loc[region]).index,df_region.loc[region]["AvgTemperature"],label = region) 
    
plt.legend()
plt.title("Growth of the average Temperture in every region over time",size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()


# In[ ]:




