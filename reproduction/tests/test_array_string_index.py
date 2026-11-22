from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

import tensorflow as tf

from kamae.spark.utils.indexer_utils import collect_labels_array
from kamae.spark.utils.user_defined_functions import indexer_udf
from kamae.keras.tensorflow.layers.string_index import StringIndexLayer

def main():
    spark = (
    SparkSession.builder
        .appName("Kamae-Array-String-Index-Test")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )

    try:
        # 1. 构造训练数据：用于收集全局词表
        train_schema = StructType([
            StructField(
                "Genres",
                ArrayType(StringType(), containsNull=False),
                nullable=False,
            )
        ])

        train_data = [
            (["Animation", "Comedy", "Children's"],),
            (["Comedy", "Romance", "Drama"],),
            (["Action", "Comedy", "Thriller"],),
        ]

        train_df = spark.createDataFrame(train_data, train_schema)

        print("\n========== 训练数据 ==========")
        train_df.show(truncate=False)

        # 2. 收集全局词表
        vocabulary = collect_labels_array(
            dataset=train_df,
            column=F.col("Genres"),
            column_datatype=ArrayType(StringType()),
            string_order_type="alphabeticalAsc",
            mask_token=None,
        )

        print("\n========== 全局词表 ==========")
        print(vocabulary)

        # 3. 构造测试数据，其中 Sci-Fi 不在训练词表中
        test_data = [
            (["Animation", "Comedy", "Sci-Fi"],),
            (["Action", "Romance", "Drama"],),
        ]

        test_df = spark.createDataFrame(test_data, train_schema)

        print("\n========== 测试数据 ==========")
        test_df.show(truncate=False)

        # 4. Spark：对数组中的每个字符串逐个索引
        spark_index_udf = F.udf(
            lambda values: [
                indexer_udf(
                    label=value,
                    labels=vocabulary,
                    mask_token=None,
                    num_oov_indices=1,
                )
                for value in values
            ],
            ArrayType(IntegerType()),
        )

        spark_result = (
            test_df
            .withColumn(
                "Spark_Indexed",
                spark_index_udf(F.col("Genres")),
            )
        )

        print("\n========== Spark 索引结果 ==========")
        spark_result.show(truncate=False)

        spark_values = [
            row["Spark_Indexed"]
            for row in spark_result.collect()
        ]

        # 5. Keras：使用相同词表进行索引
        keras_layer = StringIndexLayer(
            vocabulary=vocabulary,
            num_oov_indices=1,
            mask_token=None,
        )

        keras_inputs = tf.constant(
            [row[0] for row in test_data],
            dtype=tf.string,
        )

        keras_outputs = keras_layer(keras_inputs)

        keras_values = keras_outputs.numpy().tolist()

        print("\n========== Keras 索引结果 ==========")
        for row in keras_values:
            print(row)

        # 6. 比较 Spark 和 Keras 的输出
        print("\n========== 一致性验证 ==========")

        all_match = True

        for i, (spark_row, keras_row) in enumerate(
            zip(spark_values, keras_values)
        ):
            spark_row = [int(x) for x in spark_row]
            keras_row = [int(x) for x in keras_row]

            match = spark_row == keras_row

            print(f"第 {i + 1} 行：")
            print("Spark:", spark_row)
            print("Keras:", keras_row)
            print("是否一致:", match)

            if not match:
                all_match = False

        if all_match:
            print("\n测试通过：Spark 和 Keras 索引结果完全一致。")
        else:
            print("\n测试未通过：两端索引结果存在差异。")

        # 7. 检查 OOV 类别
        oov_index = indexer_udf(
            label="Sci-Fi",
            labels=vocabulary,
            mask_token=None,
            num_oov_indices=1,
        )

        print("\n========== OOV 检查 ==========")
        print("Sci-Fi 的 Spark OOV 索引:", oov_index)
        print("预期 OOV 索引:", 0)

        assert all_match, "Spark 和 Keras 索引结果不一致！"
        assert oov_index == 0, "OOV 索引不符合预期！"

        print("\n全部测试通过！")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()