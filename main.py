import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/train.csv")   # <-- check folder name & case!
print(df.head())                          # ✅ shows first 5 rows
