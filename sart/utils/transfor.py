import re

# 定义一个函数来提取区间字符串中的上限和下限
def extract_interval_bounds(interval_str):
    match = re.match(r'Interval\[\(([^,]+),\s([^)]+)\)\]', interval_str)
    if match:
        lower_bound = float(match.group(1))
        upper_bound = float(match.group(2))
        return lower_bound, upper_bound
    else:
        return None

# 原始区间字符串
interval_str = "Interval[(-9.241000000000001, 14.049000000000001)]"

# 提取上限和下限值
bounds = extract_interval_bounds(interval_str)

if bounds:
    lower_bound, upper_bound = bounds
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
else:
    print("Invalid interval format.")
