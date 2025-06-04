from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# def normalize_group(df: pd.DataFrame) -> pd.DataFrame:
#     def custom_normalization(x):
#         x = x.values.reshape(-1, 1)
#         scaler = MinMaxScaler()
#         x_scaled = scaler.fit_transform(x)
        
#         # Custom transformation to give higher values to smaller positions
#         x_transformed = 1 - x_scaled
        
#         return x_transformed.ravel()

#     normed = (
#         df
#         .groupby('RaceID')['LivePosition']
#         .transform(custom_normalization)
#     )
#     normed.name = 'LivePosition_norm'

#     return pd.concat([df, normed], axis=1)


def normalize_group(df: pd.DataFrame) -> pd.DataFrame:
    # 1) compute the entire normalized Series in one go
    normed = (
        df
        .groupby('RaceID')['Drv_RankScoreAdjusted']
        .transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1,1)).ravel())
        # .transform(
        #     lambda x: MinMaxScaler(feature_range=(1, 0))
        #                 .fit_transform(x.values.reshape(-1, 1))
        #                 .ravel()
        # )
    )
    normed.name = 'RankScoreAdjusted_norm'

    # 2) concat it back in one shot
    return pd.concat([df, normed], axis=1)

