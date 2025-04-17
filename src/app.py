from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
df.head()

# %%
df.to_csv("../data/raw/total_data.csv", index = False)

# %%
df.shape

# %%
df.info()

# %%
#Cuales son los duplicados
duplicates = df.duplicated()

# %%
df.duplicated(subset='id').sum()

# %%

df.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True, errors='ignore')
df.head()

# %%
print(df.select_dtypes(include=['object', 'category']))

# %%
print(df.dtypes)

# %%
df.info()

# %%
df.minimum_nights.value_counts()

# %%
df.price.value_counts()

# %%
df.room_type.value_counts()

# %%
df.neighbourhood.value_counts()

# %%
df.neighbourhood_group.value_counts()

# %%

plt.figure(figsize=(10, 6))
df['price'].hist(bins=30, edgecolor='white', color='#69b3a2')
plt.title('Price Distribution', fontsize=16, color='darkblue', fontweight='bold')
plt.xlabel('Price', fontsize=12, color='darkgreen')
plt.ylabel('Frecuency', fontsize=12, color='darkgreen')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# %%
plt.figure(figsize=(8, 5))

sns.boxplot(x='price', y='neighbourhood_group', data=df)

plt.title('Boxplot de neighbourhood por price')
plt.xlabel('price')
plt.ylabel('neighbourhood')
plt.show()
df.describe()

# %%


sns.set(style="whitegrid")  

fig, axis = plt.subplots(2, 3, figsize=(12, 8))

fig.patch.set_facecolor('pink')

colores = ['#FF7F0E', '#2CA02C', '#1F77B4', '#D62728', '#9467BD']

sns.histplot(ax=axis[0, 0], data=df, x="host_id", color=colores[0])
sns.histplot(ax=axis[0, 1], data=df, x="neighbourhood_group", color=colores[1]).set_xticks([])
sns.histplot(ax=axis[0, 2], data=df, x="neighbourhood", color=colores[2]).set_xticks([])
sns.histplot(ax=axis[1, 0], data=df, x="room_type", color=colores[3])
sns.histplot(ax=axis[1, 1], data=df, x="availability_365", color=colores[4])

fig.delaxes(axis[1, 2])

plt.tight_layout()

plt.show()

# %%
df.info()


# %% [markdown]
# Numeric Variables analysis

# %%

fig, axis = plt.subplots(4, 2, figsize=(10, 14), gridspec_kw={"height_ratios": [6, 1, 6, 1]})

colors = sns.color_palette("muted", 8)

fig.patch.set_facecolor('lightblue')

axis[0, 0].set_facecolor('lavenderblush')
axis[1, 0].set_facecolor('lavender')
axis[0, 1].set_facecolor('mintcream')
axis[1, 1].set_facecolor('mistyrose')
axis[2, 0].set_facecolor('lightcyan')
axis[3, 0].set_facecolor('honeydew')
axis[2, 1].set_facecolor('aliceblue')
axis[3, 1].set_facecolor('lavender')

sns.histplot(ax=axis[0, 0], data=df, x="price", color=colors[0])
sns.boxplot(ax=axis[1, 0], data=df, x="price", color=colors[1])

sns.histplot(ax=axis[0, 1], data=df, x="minimum_nights", color=colors[2]).set_xlim(0, 200)
sns.boxplot(ax=axis[1, 1], data=df, x="minimum_nights", color=colors[3])

sns.histplot(ax=axis[2, 0], data=df, x="number_of_reviews", color=colors[4])
sns.boxplot(ax=axis[3, 0], data=df, x="number_of_reviews", color=colors[5])

sns.histplot(ax=axis[2, 1], data=df, x="calculated_host_listings_count", color=colors[6])
sns.boxplot(ax=axis[3, 1], data=df, x="calculated_host_listings_count", color=colors[7])

plt.tight_layout()

plt.show()

# %%
sns.scatterplot(data=df, x='calculated_host_listings_count', y='price')

# %%
sns.scatterplot(data=df, x='number_of_reviews', y='price', hue='room_type', palette='Set2')


# %%
sns.scatterplot(data=df, x='availability_365', y='price', hue='room_type', palette='Paired')


# %%
sns.scatterplot(data=df, x='latitude', y='price', hue='room_type', palette='Set1')


# %%
sns.scatterplot(data=df, x='longitude', y='price', hue='room_type', palette='coolwarm')


# %%
sns.scatterplot(data=df, x='minimum_nights', y='price', hue='room_type', palette='Pastel1')

plt.show()

# %%
g = sns.FacetGrid(df, col="room_type", height=5, col_wrap=3)

g.map(sns.scatterplot, 'number_of_reviews', 'price', alpha=0.6, color='purple')

g.set_axis_labels('Number of Reviews', 'Price')

plt.show()

# %%
fig, axis = plt.subplots(figsize=(15, 7), ncols=2)

sns.barplot(ax=axis[0], data=df, x="minimum_nights", y="price", hue="minimum_nights")
axis[0].set_title('Relación entre precio y noches mínimas')

sns.barplot(ax=axis[1], data=df, x="number_of_reviews", y="price", hue="number_of_reviews")
axis[1].set_title('Relación entre precio y número de reseñas')

plt.tight_layout()


plt.show()

# %%
df[['price', 'calculated_host_listings_count']].corr()

# %%
df.groupby('calculated_host_listings_count')['price'].mean()

# %% [markdown]
# Numerical-Numerical Analysis

# %%
fig, axis = plt.subplots(4, 2, figsize=(12, 16))

sns.regplot(ax=axis[0, 0], data=df, x="minimum_nights", y="price", scatter_kws={'color': 'purple'}, line_kws={'color': 'orange'})
sns.heatmap(df[["price", "minimum_nights"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False, cmap="Blues")

sns.regplot(ax=axis[0, 1], data=df, x="number_of_reviews", y="price", scatter_kws={'color': 'teal'}, line_kws={'color': 'red'}).set(ylabel=None)
sns.heatmap(df[["price", "number_of_reviews"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1], cmap="Purples")

sns.regplot(ax=axis[2, 0], data=df, x="calculated_host_listings_count", y="price", scatter_kws={'color': 'darkblue'}, line_kws={'color': 'yellow'}).set(ylabel=None)
sns.heatmap(df[["price", "calculated_host_listings_count"]].corr(), annot=True, fmt=".2f", ax=axis[3, 0], cmap="magma").set(ylabel=None)

fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

plt.tight_layout()

plt.show()

# %% [markdown]
# Categorical-Categorical analysis

# %%
fig, axis = plt.subplots(figsize=(5, 4))

sns.countplot(data=df, x="room_type", hue="neighbourhood_group", palette="Set3")

plt.show()

# %%
df["room_type"] = pd.factorize(df["room_type"])[0]
df["neighbourhood_group"] = pd.factorize(df["neighbourhood_group"])[0]
df["neighbourhood"] = pd.factorize(df["neighbourhood"])[0]
fig, axes = plt.subplots(figsize=(15, 15))
sns.heatmap(df[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), 
            annot=True, fmt=".2f", cmap="coolwarm")

plt.tight_layout()
plt.show()

# %%
sns.pairplot(data = df)

# %%
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["price"])
plt.title("Boxplot price per night")
plt.xlabel("Price")
plt.show()

# %%
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["minimum_nights"])
plt.title("Boxplot Minimum nights")
plt.show()

# %%
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["number_of_reviews"])
plt.title("Boxplot Number of Reviews")
plt.show()


# %%
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["calculated_host_listings_count"])
plt.title("Boxplot Calculated host listing count")
plt.show()


# %% [markdown]
# Remove outliers using standar deviation method

# %%
def remove_outliers_std(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    lower_limit = mean - n_std * std
    upper_limit = mean + n_std * std
    filtered_df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    return filtered_df
columns_to_clean = ["price", "minimum_nights", "number_of_reviews", "calculated_host_listings_count"]

filtered_df = df.copy()

for col in columns_to_clean:
    filtered_df = remove_outliers_std(filtered_df, col)
print("Original shape:", df.shape)
print("Filtered shape:", filtered_df.shape)



# %% [markdown]
# Feature scaling

# %%
from sklearn.preprocessing import MinMaxScaler


numeric_columns = ["price", "minimum_nights", "number_of_reviews", 
                   "calculated_host_listings_count", "availability_365"]

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(filtered_df[numeric_columns])

scaled_df = pd.DataFrame(scaled_features, columns=numeric_columns)

filtered_df[numeric_columns] = scaled_df

filtered_df.head()

# %%
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="mean")
df_imputed = imputer.fit_transform(scaled_df)

df_imputed = pd.DataFrame(df_imputed, columns=scaled_df.columns)

numeric_columns = ["price", "minimum_nights", "number_of_reviews", 
                   "calculated_host_listings_count", "availability_365"]

scaler = MinMaxScaler()
df_imputed[numeric_columns] = scaler.fit_transform(df_imputed[numeric_columns])


X = df_imputed.drop("price", axis = 1)
y = df_imputed["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selection_model = SelectKBest(f_regression, k = 4)


selection_model.fit(X_train, y_train)


ix = selection_model.get_support()

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[ix])

X_train_sel.head()

# %%
X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)