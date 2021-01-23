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

## Day 7 : NumPy 的矩陣函式與線性代數應用

## Day 8 : NumPy 結構化陣列

---
<img width=150 src="pd.png"></img>
## Day 9 : 使用 Pandas 讀寫各種常用的檔案格式

## Day 10 : Pandas 資料索引操作 (資料過濾、選擇與合併)
