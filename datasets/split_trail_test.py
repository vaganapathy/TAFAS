# In notebook, if you only have one file:
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/vasanthakumarganapathy/vasanth/test/tsdb_greeks_test_NIFTY24104_fixed.csv")
train_df, temp = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)

for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    data.to_csv(f"/Users/vasanthakumarganapathy/vasanth/test/{name}.csv", index=False)