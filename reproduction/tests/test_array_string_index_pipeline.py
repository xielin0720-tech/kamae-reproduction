import numpy as np
import tensorflow as tf

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    StringType,
)

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers.array_string_index import (
    ArrayStringIndexTransformer,
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-Array-String-Index-Pipeline-Test")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()

    try:
        # 1. 构造测试数据
        schema = StructType([
            StructField(
                "genres",
                ArrayType(StringType(), containsNull=False),
                nullable=False,
            )
        ])

        data = [
            (["Action", "Comedy"],),
            (["Drama", "Action"],),
            (["Comedy", "Drama"],),
        ]

        df = spark.createDataFrame(data, schema)

        print("========== 原始 Spark 数据 ==========")
        df.show(truncate=False)

        # 2. 创建自定义 Transformer
        transformer = ArrayStringIndexTransformer(
            inputCol="genres",
            outputCol="genre_indices",
            inputDtype="string",
            outputDtype="bigint",
            layerName="genre_string_index",
            labelsArray=["Action", "Comedy", "Drama"],
            stringOrderType="alphabeticalAsc",
            numOOVIndices=1,
        )

        # 3. 构建 Kamae Spark Pipeline
        pipeline = KamaeSparkPipeline(
            stages=[transformer]
        )

        print("========== Pipeline FIT ==========")
        fitted_pipeline = pipeline.fit(df)

        # 4. 执行 Spark 端转换
        print("========== Spark Pipeline TRANSFORM ==========")
        spark_result = fitted_pipeline.transform(df)
        spark_result.show(truncate=False)

        spark_outputs = np.asarray(
            [
                row["genre_indices"]
                for row in spark_result.select("genre_indices").collect()
            ],
            dtype=np.int64,
        )

        print("Spark outputs:")
        print(spark_outputs)

        # 5. 使用 Pipeline 导出 Keras 模型
        input_schema = [
            {
                "name": "genres",
                "dtype": "string",
                "shape": (2,),
            }
        ]

        print("========== BUILD KERAS MODEL ==========")
        keras_model = fitted_pipeline.build_keras_model(
            input_schema=input_schema,
        )

        keras_model.summary()

        # 6. 构造相同的 Keras 输入
        keras_inputs = tf.constant(
            [
                ["Action", "Comedy"],
                ["Drama", "Action"],
                ["Comedy", "Drama"],
            ],
            dtype=tf.string,
        )

        print("========== KERAS MODEL INFERENCE ==========")
        keras_result = keras_model(
            {"genres": keras_inputs},
            training=False,
        )

        print("Keras result type:", type(keras_result))

        if isinstance(keras_result, dict):
            print("Keras result keys:", list(keras_result.keys()))
            keras_tensor = keras_result.get("genre_indices")

            if keras_tensor is None and len(keras_result) == 1:
                keras_tensor = next(iter(keras_result.values()))

            if keras_tensor is None:
                raise ValueError(
                    f"Unexpected Keras output keys: {list(keras_result.keys())}"
                )
        elif isinstance(keras_result, (list, tuple)):
            if len(keras_result) != 1:
                raise ValueError(
                    f"Expected one Keras output, got {len(keras_result)}"
                )
            keras_tensor = keras_result[0]
        else:
            keras_tensor = keras_result

        keras_outputs = np.asarray(keras_tensor.numpy())

        print("Keras outputs:")
        print(keras_outputs)

        # 7. 比较 Spark 与 Keras 输出
        print("========== SPARK / KERAS PARITY CHECK ==========")

        print("Spark outputs:")
        print(spark_outputs)

        print("Keras outputs:")
        print(keras_outputs)

        np.testing.assert_array_equal(
            spark_outputs,
            keras_outputs,
        )

        print("Pipeline integration test passed!")
        print("Spark and Keras outputs are identical.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

