import pandas as pd

df1 = pd.read_csv(r"c:\SwiftledML\data\Dataset_terbaru_real\sensor_data_lantai_1_combined (1).csv")
df2 = pd.read_csv(r"c:\SwiftledML\data\Dataset_terbaru_real\sensor_data_lantai_2_combined (1).csv")
real = pd.concat([df1, df2], ignore_index=True)
real = real[real["temperature_c"] < 50]
real.to_csv(r"c:\SwiftledML\data\real_only_lt1_lt2.csv", index=False)
print(f"{len(real)} rows saved to data/real_only_lt1_lt2.csv")
