from typing import List, Dict, Any, Tuple
import re
import polyline
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:

    result = []
    i = 0
    
    while i < len(lst):

        temp = []
        for j in range(i, min(i + n, len(lst))):
            temp.append(lst[j])
        for k in range(len(temp) - 1, -1, -1):
            result.append(temp[k])
        i += n
    
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    length_dict = {}

    for s in lst:
        length = len(s)

        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(s)

    return dict(sorted(length_dict.items()))


def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:

    def _flatten(sub_dict: Any, parent_key: str = '') -> Dict[str, Any]:

        items = []

        for k, v in sub_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        items.extend(_flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, v))
        
        return dict(items)

    return _flatten(nested_dict)

from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:

    def backtrack(start: int):

        if start == len(nums):
            result.append(nums[:])  
            return
        
        seen = set() 
        
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]

    nums.sort() 
    result = []
    backtrack(0)
    return result


def find_all_dates(text: str) -> List[str]:
   
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b', 
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b' 
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:

    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

        R = 6371000  
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0

    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:

    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    result_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) 
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            result_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j] 
    
    return result_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
  
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    

    grouped = df.groupby(['id', 'id_2'])
    
    incorrect_timestamps = pd.Series(index=pd.MultiIndex.from_product([grouped.groups.keys(), ['incorrect']]), dtype=bool)
    
    for (id_, id_2), group in grouped:

        start_dates = group['start_datetime'].dt.date.unique()
        end_dates = group['end_datetime'].dt.date.unique()

        all_days_covered = len(set(start_dates) | set(end_dates)) == 7
        
        full_day_covered = True
        for day in set(start_dates) | set(end_dates):

            day_start_times = group[(group['start_datetime'].dt.date == day)]['start_datetime']
            day_end_times = group[(group['end_datetime'].dt.date == day)]['end_datetime']
            
            if day_start_times.empty or day_end_times.empty:
                full_day_covered = False
                break

            if not ((day_start_times.min().time() <= pd.Timestamp('00:00:00').time()) and
                    (day_end_times.max().time() >= pd.Timestamp('23:59:59').time())):
                full_day_covered = False
                break

        incorrect_timestamps[(id_, id_2)] = not (all_days_covered and full_day_covered)

    return incorrect_timestamps
