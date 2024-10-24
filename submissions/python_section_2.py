import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    n = len(unique_ids)

    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        from_id = row['id_start']
        to_id = row['id_end']
        distance = row['distance']
        
        distance_matrix.at[from_id, to_id] = min(distance_matrix.at[from_id, to_id], distance)
        distance_matrix.at[to_id, from_id] = min(distance_matrix.at[to_id, from_id], distance)  # Ensure symmetry

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

