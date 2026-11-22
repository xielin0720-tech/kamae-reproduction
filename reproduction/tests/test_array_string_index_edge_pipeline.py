from pathlib import Path

import numpy as np
import tensorflow as tf

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers.array_string_index import (
    ArrayStringIndexTransformer,
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-Array-String-Index-Edge-Test")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


def create_transformer():
    return ArrayStringIndexTransformer(
        inputCol="genres",
        outputCol="genre_indices",
        inputDtype="string",
        outputDtype="bigint",
        layerName="genre_string_index",
        labelsArray=["Action", "Comedy", "Drama"],
        stringOrderType="alphabeticalAsc",
        numOOVIndices=1,
    )


def test_spark_keras_oov_parity(spark):
    """验证已知类别、未知类别和空字符串的 Spark/Keras 输出一致。"""

    schema = StructType([
        StructField(
            "genres",
            ArrayType(StringType(), containsNull=False),
            nullable=False,
        )
    ])

    rows = [
        (["Action", "Unknown"],),
        (["Comedy", ""],),
        (["Drama", "Action"],),
    ]

    df = spark.createDataFrame(rows, schema)

    pipeline = KamaeSparkPipeline(
        stages=[create_transformer()]
    )

    fitted_pipeline = pipeline.fit(df)

    # Spark 端转换
    spark_result = fitted_pipeline.transform(df)

    spark_outputs = np.asarray(
        [
            row["genre_indices"]
            for row in spark_result.select("genre_indices").collect()
        ],
        dtype=np.int64,
    )

    # 构建 Keras 模型
    input_schema = [
        {
            "name": "genres",
            "dtype": "string",
            "shape": (2,),
        }
    ]

    keras_model = fitted_pipeline.build_keras_model(
        input_schema=input_schema
    )

    keras_inputs = tf.constant(
        [
            ["Action", "Unknown"],
            ["Comedy", ""],
            ["Drama", "Action"],
        ],
        dtype=tf.string,
    )

    keras_result = keras_model(
        {"genres": keras_inputs},
        training=False,
    )

    if isinstance(keras_result, dict):
        keras_tensor = keras_result["genre_indices"]
    else:
        keras_tensor = keras_result

    keras_outputs = np.asarray(
        keras_tensor.numpy(),
        dtype=np.int64,
    )

    expected = np.asarray(
        [
            [1, 0],
            [2, 0],
            [3, 1],
        ],
        dtype=np.int64,
    )

    print("\n========== OOV SPARK / KERAS PARITY ==========")
    print("Spark outputs:")
    print(spark_outputs)
    print("Keras outputs:")
    print(keras_outputs)
    print("Expected outputs:")
    print(expected)

    np.testing.assert_array_equal(spark_outputs, expected)
    np.testing.assert_array_equal(keras_outputs, expected)
    np.testing.assert_array_equal(spark_outputs, keras_outputs)

    print("OOV parity test passed!")


def test_spark_array_edge_cases(spark):
    """检查 Spark 对空数组、NULL 数组和 NULL 数组元素的处理。"""

    schema = StructType([
        StructField(
            "genres",
            ArrayType(StringType(), containsNull=True),
            nullable=True,
        )
    ])

    rows = [
        ([],),
        (None,),
        (["Action", None],),
        (["Unknown"],),
    ]

    df = spark.createDataFrame(rows, schema)

    transformer = create_transformer()
    result = transformer.transform(df)

    actual = [
        row["genre_indices"]
        for row in result.select("genre_indices").collect()
    ]

    expected = [
        [],
        None,
        [1, 0],
        [0],
    ]

    print("\n========== SPARK ARRAY EDGE CASES ==========")
    print("Actual:", actual)
    print("Expected:", expected)

    assert actual == expected, (
        f"Spark edge-case result mismatch: "
        f"actual={actual}, expected={expected}"
    )

    print("Spark array edge-case test passed!")


if __name__ == "__main__":
    spark = create_spark_session()

    try:
        test_spark_keras_oov_parity(spark)
        test_spark_array_edge_cases(spark)

        print("\nAll edge tests passed!")

    finally:
        spark.stop()