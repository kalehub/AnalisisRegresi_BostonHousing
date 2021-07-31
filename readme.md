# Analisis Regresi : Boston Housing
_Exercise project_ implementasi Regresion Analysis menggunakan scikit-learn. Data diunduh dari situs [Kaggle](https://www.kaggle.com/vikrishnan/boston-house-prices).  
oleh: Teguh Satya  

mari berteman di [Github](https://www.github.com/kalehub)!


```python
# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Membaca dataset


```python
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('DATASET/housing.csv', header=None, delimiter=r"\s+", names=columns)
```

__Keterangan kolom:__
* CRIM : Tingkat Kriminalitas
* ZN : Land zoned
* INDUS : Proporsi bisnis non-retail dalam kota
* CHAS : Variabel dummy Sungai Charles (1 jika saluran membatasi sungai; 0 sebaliknya)
* NOX : konsentrasi oksida nitrat (bagian per 10 juta)
* RM : rata-rata jumlah kamar per hunian
* Age : proporsi unit yang ditempati pemilik yang dibangun sebelum tahun 1940
* DIS : jarak tertimbang ke lima pusat kerja Boston
* RAD : indeks aksesibilitas ke jalan raya radial
* TAX : tarif pajak properti nilai penuh per \$10.000
* PIRATIO : rasio murid-guru menurut kota
* B: 1000(Bk - 0,63)^2 di mana Bk adalah proporsi orang kulit hitam menurut kota
* LSTAT : status penduduk yang lebih rendah
* MEDV : Nilai median rumah yang ditempati pemilik di \$1000's


## Melihat dataset


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



## Melihat informasi kolom dan tipe data pada dataset


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    int64  
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    int64  
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  MEDV     506 non-null    float64
    dtypes: float64(12), int64(2)
    memory usage: 55.5 KB


## Melihat informasi statistik pada dataset


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Eksplorasi Data
Melihat informasi data dalam bentuk grafik. Hal pertama yang akan dilihat adalah distribusi data dalam atribut 'MEDV'


```python
sns.displot(df['MEDV'])
```




    <seaborn.axisgrid.FacetGrid at 0x7fde8a5fdf40>




    
![png](boston_housing_c_files/boston_housing_c_12_1.png)
    


dapat dilihat bahwa distribusi data dalam atribut 'MEDV' cukup normal dengan sedikit outliers.


```python
correlation_mat = df.corr()
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(correlation_mat,annot=True)
```




    <AxesSubplot:>




    
![png](boston_housing_c_files/boston_housing_c_14_1.png)
    


Korelasi antar atribut digambarkan dalam bentuk heatmap. Semakin "panas" korelasi antar atribut (yang dalam hal ini berwarna lebih cerah) maka semakin kuat juga nilai korelasinya secara positif. Agar proses identifikasi korelasi antar atribut dapat dilakukan dengan mudah dalam heatmap, ditambahkan angka sebagai notasi heatmap tersebut. Semakin dekat dengan angka 1, maka semakin kuat pula korelasi antar atributnya.  

Dari heatmap diatas dapat dilihat bahwa **RM merupakan atribut yang memiliki korelasi positif terbesar terhadap atribut MEDV** sementara **LSTAT merupakan atribut yang memiliki korelasi negatif terbesar terhadap MEDV**

## Melihat pengaruh atribut RM dan LSTAT dengan atribut MEDV


```python
sns.scatterplot(x=df['RM'], y=df['MEDV'],data=df)
```




    <AxesSubplot:xlabel='RM', ylabel='MEDV'>




    
![png](boston_housing_c_files/boston_housing_c_17_1.png)
    


Dapat dilihat dalam grafik scatter diatas bahwa :  

__Semakin banyak rata-rata kamar hunian (atribut RM) maka semakin tinggi pula harga rumah tersebut (atribut MEDV)__


```python
sns.scatterplot(x=df['LSTAT'], y=df['MEDV'],data=df)
```




    <AxesSubplot:xlabel='LSTAT', ylabel='MEDV'>




    
![png](boston_housing_c_files/boston_housing_c_19_1.png)
    


Dapat dilihat dalam grafik scatter diatas bahwa :  

__Semakin tinggi nilai atribut LSTAT maka harga rumah (MEDV) akan semakin rendah)__

## Pemodelan Data
Untuk memulai pemodelan, data akan diseleksi berdasarkan atribut yang telah dipilih untuk digunakan. Dalam kasus ini, atribut yang digunakan adalah RM dan LSTAT sesuai dengan hasil _correlation matrix_


```python
X = df[['RM', 'LSTAT']]
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RM</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.575</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.421</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.185</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.998</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.147</td>
      <td>5.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>6.593</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>502</th>
      <td>6.120</td>
      <td>9.08</td>
    </tr>
    <tr>
      <th>503</th>
      <td>6.976</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>504</th>
      <td>6.794</td>
      <td>6.48</td>
    </tr>
    <tr>
      <th>505</th>
      <td>6.030</td>
      <td>7.88</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 2 columns</p>
</div>




```python
y = df['MEDV']
y
```




    0      24.0
    1      21.6
    2      34.7
    3      33.4
    4      36.2
           ... 
    501    22.4
    502    20.6
    503    23.9
    504    22.0
    505    11.9
    Name: MEDV, Length: 506, dtype: float64



## Membagi dataset menjadi _Data Training_ dan _Data Testing_
Data dibagi menjadi data training dan data testing. Pada langkah ini, data training akan diambil secara random sebanyak 70% dari dataset, sedangkan data testing akan diambil sebanyak 30%


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)
```

## Pemodelan


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```




    LinearRegression()



## Membuat prediksi


```python
pred = model.predict(X_test)
pred
```




    array([36.28025974, 31.31680311, 20.34064595, 20.05835222, 29.43750447,
           30.65251222, 39.10360656, 10.70436236, 30.86911757,  8.98397456,
           27.78794289, 14.57407209, 19.19660171, 23.80576252, 23.73075217,
           19.60513187,  8.42966025, 31.3370199 , 28.00835296, 26.7059072 ,
           12.457809  , 18.49829828, 24.42204789, 31.64746687, 32.5127574 ,
           21.19959532, 27.5343067 , 21.22640793, 23.03040155, 31.03733045,
           23.9221402 , 20.72892183, 34.23426113, 36.37490772, 24.33305207,
           21.76518637, 19.81800748, 21.14376569,  6.38780592, 28.37378822,
           21.65744167, 25.54103204, 35.62144121, 12.72538325, 18.95905058,
           25.96931217, 30.75252021, 17.86202082, 27.5968919 , 29.28672731,
           32.69884285, 38.91436803, 19.2949121 , 21.65548439, 32.36795189,
           -4.60173762, 18.64832651, 16.26705808, 17.90276397, 18.81977485,
           31.85346453,  1.87870551, 13.48720801, 21.95029568, 12.68857962,
           26.61684894, 24.05655771, 19.85754479, 17.92519771, 20.33122906,
           22.75151557, 26.48069634, 20.00805237, 23.22942147, 27.15794917,
           20.18198725, 35.32534671,  6.06341837, 29.47487235, 17.50024546,
           18.26345336, 21.64018575, 28.53054706, 19.14796211,  7.73127193,
           24.13615168, 21.82450864, 28.71601837, 22.61095414, 23.46007724,
           11.55132593,  9.33804318, 26.18868367, 28.16676703,  6.46183763,
           35.08014375,  4.73209288, 30.69772665, 12.65012075, 23.13733739,
           32.59919684, 22.06206228, 25.93177452, 26.1727334 , 21.98604332,
           22.87476116, 26.71559869, 32.13967032, 37.30546722, 30.71110047,
           21.80504079, 37.34526421, 25.55731684, 21.16993168, 31.64490046,
           25.94518082, 27.84100638, 29.4388451 , 23.59460458, 25.33688181,
           21.32089115, 20.47885564, 22.78600496, 30.72489623, 27.47872671,
           23.47993203, 20.7503195 ,  3.78519868, 15.3484494 , 23.90351118,
           26.12737916, 31.45258871, 23.77286592, 20.09672118, 26.28210838,
           23.45219821, 26.0910224 , 26.08577474, 34.76464133, 24.59419526,
           16.84062057, 34.95748234, 30.73777328, 18.64536563, 29.20098188,
           23.11627926, 23.79076594, 25.01188765, 20.02852131, 26.98194717,
           19.66652126, 17.19649913])



## Scatter Plot hasil prediksi
_Scatter Plot_ ditampilkan untuk memastikan bahwa hasil prediksi terdistribusi dengan normal


```python
plt.scatter(y_test, pred)
plt.xlabel('Nilai Data Test')
plt.ylabel('Nilai Prediksi')

```




    Text(0, 0.5, 'Nilai Prediksi')




    
![png](boston_housing_c_files/boston_housing_c_31_1.png)
    


Dari grafik tersebut dapat dilihat bahwa data telah terdistribusi dengan normal secara positif

## Evaluasi Model
Pengevaluasian model regresi dengan menggunakan MAE,MSE dan RMSE


```python
from sklearn import metrics
print('***Hasil evaluasi Matriks***')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
```

    ***Hasil evaluasi Matriks***
    MAE: 4.380474756133433
    MSE: 36.54113133122373
    RMSE: 6.044926081535136

