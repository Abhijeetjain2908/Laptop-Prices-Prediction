import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv(r"C:\Users\HP\Desktop\machine learning\laptop price prediction\laptop_data.csv")
df=df.drop(columns="Unnamed: 0")
print(df)

print(df.duplicated().sum())
df=df.drop_duplicates()
print(df.isnull().sum())

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# print(df)

sns.distplot(df['Price'])
# plt.show()

df['Company'].value_counts().plot(kind='bar')
# plt.show()

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
# plt.show()

sns.distplot(df['Inches'])
# plt.show()

sns.scatterplot(x=df['Inches'],y=df['Price'])
# plt.show()

print(df['ScreenResolution'].value_counts())


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
print(df['Touchscreen'])

print(df.sample(10))

df['Touchscreen'].value_counts().plot(kind='bar')
# plt.show()

sns.barplot(x=df['Touchscreen'],y=df['Price'])
# plt.show()

df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df['Ips'].value_counts().plot(kind='bar')
# plt.show()

sns.barplot(x=df['Ips'],y=df['Price'])
# plt.show()

new = df['ScreenResolution'].str.split('x',n=1,expand=True)
df['X_res'] = new[0]
df['Y_res'] = new[1]
# print(df['X_res'])
# print(df['Y_res'])

print(df)
df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
print(df)

df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

categorical = [var for var in df.columns if df[var].dtype=='O']
numercial = [var for var in df.columns if df[var].dtype!='O']
print(df[numercial].corr()['Price'])

df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')
print(df)
numercial = [var for var in df.columns if df[var].dtype!='O']
print(df[numercial].corr()['Price'])

df.drop(columns=['ScreenResolution'],inplace=True)
df.drop(columns=['Inches','X_res','Y_res'],inplace=True)

print(df['Cpu'].value_counts())
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
print(df.head())

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return 'Intel'
    else:
        if text.split()[0] == 'Intel':
            return 'Intel'
        else:
            return 'AMD'

df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
print(df)
plt.show()

df['Cpu brand'].value_counts().plot(kind='bar')
sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns=['Cpu','Cpu Name'],inplace=True)
print(df)


df['Ram'].value_counts().plot(kind='bar')
sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['Memory'].value_counts()

df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)
 
df["first"]= new[0]
df["first"]=df["first"].str.strip()
 
df["second"]= new[1]
 
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
 
df['first'] = df['first'].str.replace(r'\D', '')
 
df["second"].fillna("0", inplace = True)
 
df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
 
df['second'] = df['second'].str.replace(r'\D', '')
 
df["first"] = df["first"].astype('int32')
df["second"] = df["second"].astype('int32')
 
df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])
 
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)

df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

df['Gpu'].value_counts()
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df['Gpu brand'].value_counts()
df = df[df['Gpu brand'] != 'ARM']
df['Gpu brand'].value_counts()
sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns=['Gpu'],inplace=True)
df['OpSys'].value_counts()

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
    
df['os'] = df['OpSys'].apply(cat_os)

df.drop(columns=['OpSys'],inplace=True)

sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.histplot(df['Weight'])
plt.show()
sns.distplot(df['Weight'])
plt.show()

sns.scatterplot(x=df['Weight'],y=df['Price'])
plt.show()