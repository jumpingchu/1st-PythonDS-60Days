# 第一屆《Python 資料科學程式馬拉松》

<img width=150 src="np.png"></img>
## Day 1 : NumPy 基本操作
* 建立陣列：`array()`, `arange()`, `linspace()`
* 建立特殊陣列：`zeros()`, `ones()`, `empty()`
* 查看陣列屬性：`shape`, `ndim`, `dtype`, `size`, `flat[index]`

## Day 2 : NumPy 陣列進階操作
### 陣列重塑
* 展開成 1 維：
    * `ravel()` 變更時原陣列也會變更
    * `flatten()` 陣列為 copy
    * param: `order='C'` for row /`'F'` for column
* 指定新形狀
    * `reshape()` 可使用模糊指定：(x, -1), 變更時原陣列也會變更
    * `resize()` 形狀錯誤時會補 0
### 軸 (axis)
* 計算方法為 **由 row 而 column**，可利用 `shape()` 來理解
* `newaxis()` 可以在指定位置新增 1 維
### 陣列合併
* `concatenate()` 除了指定軸 (預設 axis 0)，其他軸的形狀必須完全相同 (否則會 ValueError)
* `stack()` 所有維度皆需相同且會多 1 維, `hstack()`, `vstack()` 規則同 `concatenate()`

### 陣列分割
* `split()`、`hsplit()`、`vsplit()`
* 注意參數 `indices_or_sections` 的用法

### 迭代
* 以 row 為準迭代，可搭配 `flat()` 列出所有元素

### 搜尋
* `amax()`, `amin()` 用在 np 函式
* `max()`, `min()` 用在陣列物件
* `argmax()`, `argmin()` 回傳的是最大值和最小值的索引
* `np.where(a > 10, "Y", "N")` 尋找滿足條件的元素，若不設定Y/N，則要合併回傳的陣列來看索引值
* `nonzero` 等同於 `np.where(array != 0)`

### 排序
* `sort()` 回傳排序後的陣列，若是陣列物件則會 in-place
* `argsort()` 回傳排序後的陣列索引值

## Day 3 : NumPy 陣列運算及數學
* 基本運算方式同 Python 運算

## Day 4 : NumPy 陣列邏輯函式
* 函數應注意是否為 element-wise 的比較邏輯 (＝該傳入陣列還是元素)

## Day 5 : NumPy 統計函式
* 注意是否要忽略 nan，不忽略的話基本上都會優先回傳 nan
* 平均值計算可以加權，如：`np.average(a, axis=1, weights=[0.25, 0.75])`

## Day 6 : 使用 NumPy 存取各種檔案內容
* .npy 與 .npz 格式是 NumPy 的檔案格式，透過 `save()`、`savez()`、`load()` 函式進行儲存與讀取。
* 針對文字檔，可以使用 `savetxt()`、`loadtxt()` 來儲存與讀取。功能更強大的 `genfromtxt()` 則是提供更多選項在讀取檔案時進行操作

## Day 7 : NumPy 的矩陣函式與線性代數應用
* 矩陣乘積 : 點積、內積、外積、矩陣乘法
* 矩陣操作 :　跡、行列式、反矩陣、轉置、特徵值與特徵向量、秩、線性系統求解
* 特殊矩陣 : 單位矩陣 (identity)、單位矩陣 (eye)、三角矩陣、單對角陣列、上三角矩陣、下三角矩陣
* 矩陣分解 : Cholesky、QR、SVD

## Day 8 : NumPy 結構化陣列
* 資料型別常在陣列中用到，NumPy 的 dtype 彈性很大，並且可以與 Python 資料型別交互使用
* NumPy 陣列也可以儲存複合式資料，也就是包含不同資料型別的元素。這就是結構化陣列 (Structured Arrays) 的功能，進行後續的資料存取及處理。
![dtype_ref](Day008/dtype%20對照表.png)

---

<img width=150 src="pd.png"></img>

## Day 9 : 使用 Pandas 讀寫各種常用的檔案格式
* 讀寫 csv: `read_csv()`, `to_csv()`
* 讀寫 excel: `read_excel()`, `to_excel()`
* 讀寫 json: `read_json()`, `to_json()`
* 讀寫 SQL 資料庫: `io.sql.read_sql()`, `to_sql()`

## Day 10 : Pandas 資料索引操作 (資料過濾、選擇與合併)
* 指定欄位名稱當做索引: `.set_index()` 
* 對欄位名稱進行重新命名: `.rename(column={'old_name': 'new_name'})`
* 增加欄位: `['new_col_name']`, `.insert()`
* 刪除欄位: `del`, `.pop()`, `.drop()`
* 增加列資料: `.append()`

## Day 11 : Pandas 類別資料、缺失值處理
### 類別資料
* 順序性的類別資料，需要有順序性的 encoding 方法，可以使用 sklearn 中的 `LabelEncoder()`
* 對於一般性的類別資料，則不需要有順序的編碼，可以使用 pandas 中的 `get_dummies()`
### 缺失值補值 
* `fillna()`
    1. 補定值
    2. 補平均值 `mean()` 或中位數 `median()`
    3. `method='ffill'`(補前值) 或 `'bfill'`（補後值）
* 內插法補值 `interpolate()`

## Day 12 : Pandas 常見圖表程式設計
### 折線圖
* 適用：會隨時間變動的值

        .plot()
### 長條圖
* 適用：不同種類資料，在不同時間點的變化

        .plot.bar(stacked=False)

### 箱型圖
* 適用：完整呈現數值分布的統計圖表

        .boxplot()

### 散佈圖
* 適用：呈現相關數值間的關係

        .plot.scatter(x, y)

## Day 13 : Pandas 統計函式使用教學
### 相關係數
1. 介於 –1 與 +1 之間，即 –1 ≤ r ≤ +1
    * r > 0 時，表示兩變數正相關
    * r < 0 時，兩變數為負相關
    * r = 0 時，表示兩變數間無線性相關
2. 一般可按三級劃分：
    * | r | < 0.4 為低度線性相關
    * 0.4 ≤ | r | < 0.7 為顯著性相關
    * 0.7 ≤ | r | < 1 為高度線性相關
3. ```python
    pandas.DataFrame.corr()
    pandas.Series.corr()
    ```

## Day 14 : 用 Pandas 撰寫樞紐分析表
* 索引轉欄位 `.unstack()`
* 欄位轉索引 `.stack()`  (注意都是由最外層開始轉換)
* 欄位名稱轉為欄位值 `.melt()`，其中參數:
    * `id_vars`：不需要被轉換的列名
    * `value_vars`：需要轉換的列名，如果剩下的列全部都要轉換，就不用寫了
* 重新組織資料 `.pivot()`，其中參數
    * `index`：新資料的索引名稱 
    * `columns`：新資料的欄位名稱
    * `values`：新資料的值名稱

## Day 15 : Split-Apply-Combine Strategy (GroupBy)
* `.groupby().agg()` 可以同時針對多個欄位做多個分析

        df.groupby(['sex', 'class']).agg(['mean', 'max'])

## Day 16 : Pandas 時間序列
* 控制時間長度的函數 `.to_period()`，參數 `freq` 代表時間頻率(Y：年 / M：月 / W：週 / D：日 / H：小時)
* 利用 `resample()` 更改時間頻率，如年轉成季 `resample('Q')`
* 移動（shifting）指的是沿著時間軸將資料前移或後移
        
        .shift(periods=1, freq=None)

* 時間需要使用 `pd.Timestamp()` 做設定
    * 例如：
    
            pd.Timestamp(2021, 2, 2)

    * 可以直接加時間或是計算時間差距

* 時間轉字串 

        date.strftime('%Y-%m-%d')

* 字串轉時間

        pd.to_datetime(str_date)

* 計算工作日 

        pd.offsets.BDay()

## Day 17 : Pandas 效能調校
* 三個加速方法
    * 讀取資料型態選最快速的 (可先存為 pkl 檔 `to_pickle()`，減少之後每次開啟所花費的時間)
    * 多使用內建函數 (如 `agg()`, `transform()`...)
    * 向量化的資料處理 (如 `isin()`...)
* 欄位的型態降級有助於減少記憶體佔用空間

---

<img width=150 src="images/matplotlib.svg"></img><br>
<img width=150 src="images/seaborn.png"></img><br>
<img width=150 src="images/bokeh.png"></img><br>

## Day 18 : Python 資料視覺化工具與常見統計圖表介紹
* matplotlib
* seaborn
* bokeh
* basemap

## Day 19 : 使用 Matplotlib 繪製各種常用圖表
* 建議以下步驟學習如何使用 Matplotlib：
    1. 學習 Matplotlib 的基本術語, 具體來說就是什麼是 Figure 和 Axes
    2. 一直使用面向對象的介面，養成習慣
    3. 用基礎的 pandas 繪圖開始可視化
    4. 使用 seaborn 進行稍微複雜的數據可視化
    5. 使用 Matplotlib 自訂 pandas 或 seaborn 視覺化

![figure_anatomy](images/anatomy_of_a_figure.png)
![figure_anatomy](images/plot_customization.png)

## Day 20 : 使用 Seaborn 進行資料視覺化
### 樣式
* 設定圖形樣式
    
        sns.set_style(“whitegrid”)

* 五種預設：darkgrid, whitegrid, dark, white, ticks

### 聚合和表示不確定性
* 對於較大的數據是通過繪製標準差來表示每個時間點的分佈，而不是信心區間

        sns.relplot(x, y, ci="sd")

* 語義映射繪製數據子集

        sns.relplot(x, y, hue="region", style="event")

### 可視化線性關係

    sns.regplot(x="total_bill", y="tip", data=tips)

