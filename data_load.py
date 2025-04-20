import pandas as pd
import os
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns  
from matplotlib import rcParams
import plotly.express as px
from pathlib import Path
import json

# 提取每条购买记录中的 payment_status
def extract_payment_status(purchase_history):
    try:
        # 解析 purchase_history 中的 JSON 字符串
        purchase_data = json.loads(purchase_history)
        return purchase_data.get('payment_status', None)  # 获取 payment_status 字段
    except Exception as e:
        return None  # 如果解析失败，返回 None
    
# 提取每条购买记录中的 avg_price 和 items
def extract_purchase_info(purchase_history):
    try:
        # 解析 purchase_history 中的 JSON 字符串
        purchase_data = json.loads(purchase_history)
        avg_price = purchase_data.get('avg_price', None)  # 获取 avg_price 字段
        items = purchase_data.get('items', [])  # 获取 items 字段（列表）
        item_count = len(items)  # 计算 items 数量
        return avg_price, item_count
    except Exception as e:
        return None, None  # 如果解析失败，返回 None
    
# 计算每个购买记录的总消费
def calculate_total_spending(purchase_history):
    # 解析 purchase_history 为字典
    purchase_data = json.loads(purchase_history)
    avg_price = purchase_data['avg_price']
    num_items = len(purchase_data['items'])
    total_spending = avg_price * num_items
    return total_spending

# 提取类别和avg_price
def extract_category_and_avg_price(purchase_history):
    try:
        # 将字符串转换为 JSON 对象
        purchase_data = json.loads(purchase_history)
        # 提取 category 和 avg_price
        category = purchase_data.get("categories", "")
        avg_price = purchase_data.get("avg_price", 0)
        return category, avg_price
    except Exception as e:
        return None, None  # 如果解析失败，返回 None
    
# 解析 purchase_history 列并提取 category 信息
def extract_category(purchase_history):
    try:
        # 将 purchase_history 字符串转换为字典
        purchase_data = json.loads(purchase_history)
        return purchase_data.get("categories", "")
    except Exception as e:
        return None  # 如果解析失败，返回 None


pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_colwidth', None)  # 显示列内容的最大宽度，None表示没有限制

# 设置支持中文的字体，解决中文显示问题
rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体，支持中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

start_time = time.time()

# 定义数据目录
data_directory = 'C:\\datasets\\data_mini\\10G_data_new'
# 获取所有parquet文件
parquet_files = glob.glob(os.path.join(data_directory, "*.parquet"))

# 读取所有Parquet文件并合并为一个DataFrame
# df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

# 需要读取的字段
columns_to_read = ['purchase_history']
# 用于存储最终合并后的数据
all_data = []
for file in parquet_files:
    # 读取指定字段
    df = pd.read_parquet(file, columns=columns_to_read)
    # 数据清理
    # df_cleaned = df[df['gender'].isin(['男', '女'])]
    # 将读取的数据添加到列表
    all_data.append(df)
# 拼接所有数据
df = pd.concat(all_data, ignore_index=True)

# 对每条记录应用提取函数，创建新的 'payment_status' 列
# df['payment_status'] = df['purchase_history'].apply(extract_payment_status)

# 对每条记录应用提取函数，创建新的 'avg_price' 和 'item_count' 列
# df[['avg_price', 'item_count']] = df['purchase_history'].apply(lambda x: pd.Series(extract_purchase_info(x)))

# 应用函数计算总消费
# df['total_spending'] = df['purchase_history'].apply(calculate_total_spending)

# 提取所有类别并创建一个新的 'category' 列
df['category'] = df['purchase_history'].apply(extract_category)

# 应用函数到 df 的 purchase_history 列
# df[['category', 'avg_price']] = df['purchase_history'].apply(lambda x: pd.Series(extract_category_and_avg_price(x)))

# file_path = '/data4/longtengyu/datasets/data_mini/30G_data_new/part-00000.parquet'
# df = pd.read_parquet(Path(file_path))

#查看数据的基本信息
# df.info()

# 查看数据的一些基本统计信息
# print(df.describe())

# 查看数据类型
# print(df.dtypes)

# print("=== 前5行数据 ===")
# print(df.head(5))

# print("=== purchase_history前5行数据 ===")
# print(df['purchase_history'].head(5))

# print("=== login_history前5行数据 ===")
# print(df['login_history'].head(5))

# 数据质量评估
# missing_stats = df.isnull().sum()
# print("缺失值统计:")
# print(missing_stats.to_markdown())

# 年龄分组配置
# bins = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
# labels = ['11-20', '21-30', '31-40', '41-50', 
#           '51-60', '61-70', '71-80', '81-90', '91-100']

# 生成年龄分组
# df['age_group'] = pd.cut(
#     df['age'],
#     bins=bins,
#     labels=labels,
#     right=False  # 左闭右开区间，匹配原始逻辑
# )

# 统计分组人数
# age_counts = df['age_group'].value_counts().sort_index()

# 可视化
# 年龄
# plt.figure(figsize=(12, 6))
# sns.barplot(x=age_counts.index, y=age_counts.values) #sns.barplot
# plt.title('age_distribution_chart')
# plt.xlabel('age')
# plt.ylabel('user_count')
# plt.savefig('age_distribution_chart.png')
# plt.show()
#性别
# gender_count = df['gender'].value_counts()
# # 绘制饼图
# plt.figure(figsize=(6, 6))  # 设置画布大小
# gender_count.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
# plt.title('gender_distribution_chart')  # 图表标题
# plt.ylabel('')  # 去掉y轴标签
# plt.savefig('gender_distribution_chart.png')
# plt.show()

# # 计算每种支付状态的频次
# payment_status_count = df['payment_status'].value_counts()
# # 绘制饼图
# plt.figure(figsize=(6, 6))  # 设置画布大小
# payment_status_count.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
# plt.title('支付状态分布')  # 图表标题
# plt.ylabel('')  # 去掉y轴标签
# # 保存饼图为 PNG 文件
# plt.savefig('payment_status_pie_chart.png', format='png', dpi=300)

# 设置图表样式
# sns.set_theme(style="whitegrid")
# # 绘制箱型图
# plt.figure(figsize=(12, 8))  # 设置画布大小
# sns.boxplot(x='country', y='income', data=df, palette='Set2')
# # 设置标题和标签
# plt.title('收入与国家之间的关系', fontsize=16)
# plt.xlabel('国家', fontsize=14)
# plt.ylabel('收入', fontsize=14)
# # 旋转x轴标签，防止重叠
# plt.xticks(rotation=45)
# # 保存图形为文件（可以修改路径和文件名）
# plt.tight_layout()
# plt.savefig('income_vs_country.png')  # 保存为 PNG 格式
# plt.show()  # 显示图形
#散点图
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x='country', y='income', data=df, hue='country', palette='Set2')
# plt.title('收入与国家之间的关系', fontsize=16)
# plt.xlabel('国家', fontsize=14)
# plt.ylabel('收入', fontsize=14)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('income_vs_country.png')
# plt.show()

# # 绘制年龄与收入的关系图（散点图）
# plt.figure(figsize=(10, 6))  # 设置图形大小
# sns.scatterplot(x='age', y='income', data=df)  # 绘制散点图
# # 设置标题和标签
# plt.title('age_income_relationship')
# plt.xlabel('age')
# plt.ylabel('income')
# # 保存图像
# plt.savefig('age_income_relationship.png')
# # 显示图像
# plt.show()

# 查看所有唯一的类别
unique_categories = df['category'].unique()
# 打印唯一类别
print("Unique categories in purchase_history:")
print(unique_categories)
# 统计每个 category 的出现次数
category_counts = df['category'].value_counts()
# 绘制饼图
plt.figure(figsize=(8, 8))
category_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3')
# 设置标题
plt.title('Purchase History Categories Distribution')
# 保存图像
plt.savefig('purchase_history_categories_pie_chart.png')
# 显示图像
plt.show()

# # 使用 seaborn 绘制类别和 avg_price 的关系图
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='category', y='avg_price', data=df)
# # 设置标题和标签
# plt.title('类别与平均价格的关系图')
# plt.xticks(rotation=45)  # 旋转 x 轴标签
# plt.tight_layout()
# # 保存图像
# plt.savefig('category_avg_price_relationship.png')
# # 显示图像
# plt.show()

# # 设置图表样式
# sns.set_theme(style="whitegrid")
# # 绘制直方图或 KDE 图
# plt.figure(figsize=(12, 8))
# sns.histplot(df['total_spending'], kde=True, color='blue', bins=30)
# # 设置标题和标签
# plt.title('总消费分布图', fontsize=16)
# plt.xlabel('总消费金额', fontsize=14)
# plt.ylabel('频率', fontsize=14)
# # 保存图形为文件（可以修改路径和文件名）
# plt.tight_layout()
# plt.savefig('total_spending_distribution.png')  # 保存为 PNG 格式
# # 显示图形
# plt.show()

load_time = time.time() - start_time
print(f"类别分布: {load_time:.2f}秒")