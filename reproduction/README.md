# Kamae 论文复现说明文档

## 1. 项目概述

### 1.1 复现目标

本项目基于论文 *Kamae: Bridging Spark and Keras for Seamless Machine Learning Preprocessing*，复现论文中使用 MovieLens 1M 数据集构建 Spark 数据预处理 Pipeline，并将训练完成的预处理流程转换为 Keras 模型的过程。

本阶段的主要目标包括：

1. 配置 Kamae 项目的运行环境。
2. 准备并加载 MovieLens 1M 数据集。
3. 根据论文 Listing 1 实现 Spark 数据预处理 Pipeline。
4. 将拟合后的 Spark Pipeline 转换为 Keras 预处理模型。
5. 验证 Spark 端与 Keras 端的预处理结果是否一致。

本阶段主要关注数据预处理流程的复现及跨框架一致性，不涉及论文全部下游模型训练实验和最终推荐效果指标的复现。

### 1.2 复现环境

本项目在 Windows 环境下完成开发和验证，主要运行环境如下：

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.10 |
| Python 环境管理 | Conda |
| Conda 环境名称 | `kamae-repro` |
| JDK | 17 |
| Hadoop | 3.5.0 |
| Apache Spark | 4.2.1 |
| TensorFlow | 2.21.0 |
| Keras | 3.12.4 |
| 数据集 | MovieLens 1M |
| Spark 运行模式 | `local[*]` |

Python 及相关依赖通过项目根目录下的 `environment.yml` 进行配置。JDK、Hadoop 和 Windows 下的 `winutils.exe` 需要单独安装和配置。

上述版本为当前开发环境使用的版本，建议组员尽量保持一致，以减少环境差异导致的运行问题。

### 1.3 项目目录结构

本阶段涉及的主要文件如下：

```text
repository/
│
├── src/
│   └── kamae/
│       └── spark/
│           └── transformers/
│               └── array_string_index.py
│
├── reproduction/
│   ├── scripts/
│   │   ├── run_movielens_pipeline.py
│   │   └── verify_movielens_parity.py
│   │
│   ├── preprocessing/
│   │   └── movielens_preprocessing.py
│   │
│   ├── tests/
│   │
│   └── outputs/
│       ├── spark_output.parquet
│       └── preprocessing_model.keras
│
└── ...
```

其中：

- `run_movielens_pipeline.py`：执行 MovieLens 数据加载、Spark Pipeline 构建、拟合、数据转换和 Keras 模型导出。
- `verify_movielens_parity.py`：验证 Spark 与 Keras 两端的预处理结果是否一致。
- `array_string_index.py`：实现针对字符串数组的逐元素索引转换。
- `spark_output.parquet`：保存 Spark 端预处理后的数据。
- `preprocessing_model.keras`：保存从 Spark Pipeline 转换得到的 Keras 模型。

注：reproduction/outputs/ 中的文件为项目运行后生成的实验产物，已加入 .gitignore，不会随 GitHub 仓库下载。组员需要运行主 Pipeline 脚本自行生成。

## 2. 数据集准备

### 2.1 MovieLens 1M 数据集

本次复现使用 MovieLens 1M 数据集。数据集包含用户信息、电影信息和用户评分记录。

本次实际加载的数据量如下：

| 数据内容 | 数量 |
|---|---:|
| 用户记录 | 6,040 |
| 电影记录 | 3,883 |
| 评分记录 | 1,000,209 |
| 关联后的数据记录 | 1,000,209 |

数据预处理过程中，将用户信息、电影信息与评分记录进行关联，形成用于后续 Pipeline 拟合和转换的训练数据集 `train_ml`。

### 2.2 输入特征

本次复现主要使用以下四项输入特征：

| 特征名称 | 含义 | 原始数据类型 |
|---|---|---|
| UserID | 用户编号 | String |
| MovieID | 电影编号 | String |
| Occupation | 用户职业类别 | String |
| Genres | 电影类型 | String |

其中，Genres 字段可能包含多个电影类型，原始数据使用 `|` 分隔，例如：

```text
Action|Adventure|Comedy|Romance
```

因此，需要先将其转换为固定长度的字符串数组，再执行逐元素索引。

### 2.3 数据集下载与存放

本项目使用 MovieLens 1M 数据集，组员需要自行下载数据集。

官方下载地址：

https://grouplens.org/datasets/movielens/1m/

下载后解压，将其中的 `users.dat`、`movies.dat` 和 `ratings.dat` 三个文件放入以下目录：

```text
reproduction/
├── data/
│   └── ml-1m/
│       ├── users.dat
│       ├── movies.dat
│       └── ratings.dat
└── preprocessing/
    └── movielens_preprocessing.py
```

注意：

1. 请使用 MovieLens 1M 数据集，不要使用其他版本。
2. 数据文件必须放置在 `reproduction/data/ml-1m/` 目录下。
3. `reproduction/data/` 已加入 `.gitignore`，因此数据集不会随 GitHub 仓库一起下载，需要组员自行准备。
4. 项目通过 `Path(__file__).resolve()` 动态定位数据集目录，不依赖开发者电脑上的固定磁盘路径。

## 3. Spark 预处理 Pipeline 实现

### 3.1 Pipeline 整体结构

根据论文 Listing 1，本次复现构建了包含五个处理阶段的 Spark Pipeline。

处理流程如下：

```text
                  MovieLens 1M
                       |
                 原始输入数据
                       |
        +--------------+--------------+
        |              |              |
      UserID         MovieID      Occupation
        |              |              |
   HashIndex       StringIndex     OneHotEncode
        |              |              |
        +--------------+--------------+
                       |
                     Genres
                       |
              StringToStringList
                       |
                固定长度数组
                       |
             ArrayStringIndex
                       |
                  预处理结果
                       |
                Spark Pipeline
                       |
                 Keras Model
```

### 3.2 UserID 哈希索引

使用 `HashIndexTransformer` 对 UserID 进行哈希索引。

主要参数：

| 参数 | 配置 |
|---|---|
| inputCol | `UserID` |
| outputCol | `UserID_indexed` |
| inputDtype | `string` |
| numBins | `10000` |
| layerName | `user_hash_indexer` |

该阶段将用户标识映射为固定范围内的整数索引，避免直接使用原始用户编号。

### 3.3 MovieID 字符串索引

使用 `StringIndexEstimator` 对 MovieID 进行字符串索引。

主要参数：

| 参数 | 配置 |
|---|---|
| inputCol | `MovieID` |
| outputCol | `MovieID_indexed` |
| inputDtype | `string` |
| stringOrderType | `frequencyDesc` |
| numOOVIndices | `1` |
| layerName | `movie_id_string_indexer` |

其中，`frequencyDesc` 表示按照标签出现频率降序排列；`numOOVIndices=1` 表示预留一个未登录类别索引。

### 3.4 Occupation 独热编码

使用 `OneHotEncodeEstimator` 对 Occupation 特征进行独热编码。

主要参数：

| 参数 | 配置 |
|---|---|
| inputCol | `Occupation` |
| outputCol | `Occupation_indexed` |
| inputDtype | `string` |
| stringOrderType | `frequencyDesc` |
| numOOVIndices | `1` |
| dropUnseen | `True` |
| layerName | `occupation_one_hot_encoder` |

该阶段将职业类别转换为数值向量，便于后续模型使用。

### 3.5 Genres 字符串拆分

由于 Genres 字段可能包含多个电影类型，因此首先使用 `StringToStringListTransformer` 将字符串拆分为固定长度的数组。

主要参数：

| 参数 | 配置 |
|---|---|
| inputCol | `Genres` |
| outputCol | `Genres_split` |
| inputDtype | `string` |
| outputDtype | `string` |
| separator | `|` |
| listLength | `6` |
| defaultValue | `PADDED` |
| layerName | `genres_split_to_array_transform` |

例如：

```text
原始值：
Action|Adventure|Comedy|Romance

转换结果：
[Action, Adventure, Comedy, Romance, PADDED, PADDED]
```

通过固定长度处理，使不同电影类型数量的记录能够形成统一的数据结构。

### 3.6 Genres 数组逐元素索引

论文中的 Genres 处理流程需要对字符串数组中的每个元素进行索引。

但当前使用的 Kamae 版本中，标准 `StringIndexTransformer` 主要处理标量字符串输入，不能直接满足本次数组逐元素索引的需求。

因此，本次复现实现了自定义 `ArrayStringIndexTransformer`，用于对 `Genres_split` 中的每个元素分别进行索引。

主要参数：

| 参数 | 配置 |
|---|---|
| inputCol | `Genres_split` |
| outputCol | `Genres_indexed` |
| inputDtype | `string` |
| outputDtype | `int` |
| stringOrderType | `frequencyDesc` |
| numOOVIndices | `1` |
| maskToken | `PADDED` |
| layerName | `genres_string_indexer` |

Genres 词表不再采用手动硬编码的 alphabetical 顺序，而是使用 Kamae 提供的 `collect_labels_array()` 函数，从训练数据中统计标签并按照出现频率降序生成词表。

主要实现逻辑：

```python
genres_split_df = genres_splitter.transform(train_ml)

genres_vocabulary = collect_labels_array(
    dataset=genres_split_df,
    column=F.col("Genres_split"),
    column_datatype=genres_split_df.schema["Genres_split"].dataType,
    string_order_type="frequencyDesc",
    mask_token="PADDED",
)
```

其中：

- 使用训练数据生成 Genres 词表。
- 统计标签出现频率。
- 按照 `frequencyDesc` 排序。
- 排除 `PADDED` 填充标记。
- 将生成的词表传递给 `ArrayStringIndexTransformer`。

本次实际生成的 Genres 词表为：

| 索引 | Genres |
|---:|---|
| 0 | PADDED（mask） |
| 1 | OOV（预留） |
| 2 | Comedy |
| 3 | Drama |
| 4 | Action |
| 5 | Thriller |
| 6 | Sci-Fi |
| 7 | Romance |
| 8 | Adventure |
| 9 | Crime |
| 10 | Horror |
| 11 | Children's |
| 12 | War |
| 13 | Animation |
| 14 | Musical |
| 15 | Mystery |
| 16 | Fantasy |
| 17 | Western |
| 18 | Film-Noir |
| 19 | Documentary |

注意：上述实际词表中，索引 2 至 19 对应 18 个有效 Genres 标签；索引 0 和 1 分别用于 mask 与 OOV。

## 4. Keras 预处理模型构建

### 4.1 模型构建方式

完成 Spark Pipeline 的拟合后，使用 Kamae 提供的 `build_keras_model()` 方法，将拟合后的预处理流程转换为 Keras 模型。

当前版本使用：

```python
keras_model = fitted_pipeline.build_keras_model(
    input_schema=input_schema,
)
```

与论文示例中的 `tf_input_schema` 参数名不同，当前版本使用 `input_schema`。这是 API 版本差异，需要按照当前项目实际接口调用。

### 4.2 Keras 输入配置

本次最终使用的输入配置如下：

```python
input_schema = [
    {
        "name": "UserID",
        "dtype": "string",
        "shape": (1,),
    },
    {
        "name": "MovieID",
        "dtype": "string",
        "shape": (1,),
    },
    {
        "name": "Occupation",
        "dtype": "string",
        "shape": (1,),
    },
    {
        "name": "Genres",
        "dtype": "string",
        "shape": (1,),
    },
]
```

其中，四个输入特征均采用字符串类型，与 Spark Pipeline 中的输入数据类型相匹配。

经过源码核对，当前 Kamae 版本的 `build_keras_inputs()` 直接执行：

```python
input_layer = keras.layers.Input(**conf)
```

因此，`shape` 会直接传递给 Keras Input。

最终采用 `(1,)`，对应 Keras 实际输入形状 `(None, 1)`，与论文 Listing 1 中给出的输入 shape 一致。

### 4.3 模型导出

模型构建成功后，将其保存为：

```text
reproduction/outputs/preprocessing_model.keras
```

该模型包含与 Spark Pipeline 对应的预处理层，可用于后续 Keras 端的数据预处理。

## 5. 与论文 Listing 1 的差异及适配

本次复现以论文 Listing 1 为主要实现依据，同时考虑当前 Kamae 代码版本的实际接口和功能支持情况。

| 对照项目 | 论文 Listing 1 | 当前复现实现 | 说明 |
|---|---|---|---|
| UserID 处理 | HashIndexTransformer | HashIndexTransformer | 核心参数一致 |
| MovieID 处理 | StringIndexEstimator | StringIndexEstimator | 核心参数一致 |
| Occupation 处理 | OneHotEncodeEstimator | OneHotEncodeEstimator | 核心参数一致 |
| Genres 拆分 | StringToStringListTransformer | StringToStringListTransformer | 参数一致 |
| Genres 数组索引 | StringIndexEstimator | ArrayStringIndexTransformer | 适配数组逐元素索引 |
| Genres 词表 | frequencyDesc | collect_labels_array 动态生成 | 不再使用手动 alphabetical 词表 |
| 模型构建参数 | tf_input_schema | input_schema | 当前版本 API 差异 |
| 输入 shape | `(1,)` | `(1,)` | 最终已对齐 |
| 输入 dtype | 示例中前三项存在 int32 配置 | string | 与 Pipeline 输入类型匹配 |

需要特别说明：

1. 自定义 `ArrayStringIndexTransformer` 是为了弥补当前版本标准 Transformer 对字符串数组逐元素索引支持不足的问题。
2. Genres 词表采用 Kamae 原生工具函数生成，避免手动维护词表。
3. 模型构建 API 参数名按照当前版本源码进行适配。
4. 论文 Listing 1 中 `inputDtype="string"` 与 `input_schema` 前三项 `dtype="int32"` 存在表面上的类型不一致。本次实现保留与实际 Pipeline 输入匹配的字符串类型，没有直接照搬可能产生类型冲突的配置。

此外，`collect_labels_array()` 对同频标签没有显式指定二级排序规则，因此同频标签之间的顺序不应被视为固定不变。

## 6. 运行与验证

### 6.1 项目获取与环境配置

首先，从 GitHub 克隆项目：

```powershell
git clone https://github.com/xielin0720-tech/kamae-reproduction.git
```

进入项目根目录：

```powershell
cd .\kamae-reproduction
```

创建 Conda 环境：

```powershell
conda env create -f environment.yml
```

环境创建完成后，激活环境：

```powershell
conda activate kamae-repro
```

**注意：**

1. 执行上述命令前，需要确保电脑已经安装 Conda。
2. `environment.yml` 负责配置 Python 及相关 Python 依赖，不会自动安装 JDK 和 Hadoop。
3. 运行 Spark 前，需要单独安装 JDK 17，并配置 `JAVA_HOME`。
4. Windows 环境下还需要准备 Hadoop 相关文件，配置 `HADOOP_HOME`，并确保 Hadoop 的 `bin` 目录中包含 `winutils.exe`。
5. JDK 和 Hadoop 的安装路径根据组员自己的电脑实际情况配置，不需要与开发者电脑的安装路径相同。

### 6.2 运行环境检查

激活 Conda 环境后，可以通过以下命令检查基础运行环境：

```powershell
python --version
java -version
hadoop version
```

其中：

- `python --version`：检查 Python 版本是否为 3.10。
- `java -version`：检查 JDK 是否正确安装。
- `hadoop version`：检查 Hadoop 是否能够正常运行。

如果上述命令无法正常执行，应先检查 Conda 环境、`JAVA_HOME`、`HADOOP_HOME` 和系统 `Path` 环境变量配置。

确认 MovieLens 数据集已放置在指定目录后，再执行后续 Pipeline。

### 6.3 执行主 Pipeline

运行：

```powershell
python .\reproduction\scripts\run_movielens_pipeline.py
```

脚本依次完成：

1. 创建 Spark Session。
2. 加载 MovieLens 1M 数据。
3. 构建预处理 Pipeline。
4. 拟合 Pipeline。
5. 使用拟合后的 Pipeline 转换数据。
6. 保存 Spark 预处理结果。
7. 构建 Keras 预处理模型。
8. 保存 Keras 模型。

### 6.4 执行 Parity 验证

运行：

```powershell
python .\reproduction\scripts\verify_movielens_parity.py
```

验证 Spark 与 Keras 两端对应的四项预处理输出。

最终验证结果：

```text
PASS: All four preprocessing outputs match.
```

该结果说明，在当前验证脚本所使用的数据和测试条件下，Spark 与 Keras 两端的四项预处理结果一致。

*****注：这个检验脚本只检验了128条数据 全部数据的检验要靠C同学完成！！！！！！！！！

### 6.5 验证结论

本次复现已完成以下验证：

- MovieLens 1M 数据能够正常加载。
- Spark Pipeline 能够正常构建、拟合和转换数据。
- 五个预处理阶段均能够正常执行。
- Genres 词表能够按照 frequencyDesc 生成。
- Keras 预处理模型能够成功构建和保存。
- Spark 与 Keras 的四项预处理输出通过一致性验证（128条）。
- 最终输入 shape 已与论文 Listing 1 对齐。

## 7. 最终产物

本阶段主要产物如下：

| 产物 | 路径 | 用途 |
|---|---|---|
| 主复现脚本 | `reproduction/scripts/run_movielens_pipeline.py` | 完整执行数据预处理与模型导出 |
| Parity 验证脚本 | `reproduction/scripts/verify_movielens_parity.py` | 验证 Spark/Keras 输出一致性 |
| 自定义 Transformer | `src/kamae/spark/transformers/array_string_index.py` | 数组元素逐项索引 |
| Spark 输出数据 | `reproduction/outputs/spark_output.parquet` | 保存预处理结果 |
| Keras 模型 | `reproduction/outputs/preprocessing_model.keras` | 保存预处理模型 |

## 8. 本阶段工作总结

本阶段完成了 Kamae 项目在 Windows 环境下的配置，以及基于 MovieLens 1M 数据集的预处理 Pipeline 复现。

在实现过程中，针对当前 Kamae 版本对字符串数组索引支持不足的问题，实现了自定义 `ArrayStringIndexTransformer`，并使用 Kamae 原生词表统计函数按照 `frequencyDesc` 生成 Genres 词表。

随后，完成了 Spark Pipeline 到 Keras 预处理模型的转换与导出，并通过 Parity 验证确认两端四项预处理输出一致。

因此，本阶段已经完成论文 Listing 1 所涉及的 MovieLens 预处理流程复现、必要的版本适配和跨框架一致性验证，可为后续下游模型训练与实验提供预处理基础。

**复现范围说明：** 本阶段验证的是预处理 Pipeline 及其 Spark/Keras 一致性，不代表论文全部实验、下游模型训练或推荐性能指标已经完成复现。