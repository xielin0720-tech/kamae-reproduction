
import shutil
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

from kamae.spark.pipeline.pipeline_model import KamaeSparkPipelineModel
from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.transformers.array_string_index import (
    ArrayStringIndexTransformer,
)


# ============================================================
# 1. 基础配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "reproduction"
    / "results"
    / "serialization_test"
)

PIPELINE_PATH = RESULT_DIR / "spark_pipeline_model"
KERAS_MODEL_PATH = RESULT_DIR / "preprocessing_model.keras"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-Array-String-Index-Serialization-Test")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


# ============================================================
# 2. 构造测试数据
# ============================================================

def create_test_dataframe(spark):

    schema = StructType([
        StructField(
            "genres",
            ArrayType(StringType(), containsNull=False),
            nullable=False,
        )
    ])

    rows = [
        (["Action", "Comedy"],),
        (["Drama", "Action"],),
        (["Comedy", "Drama"],),
        (["Unknown", "Action"],),
    ]

    return spark.createDataFrame(rows, schema)


# ============================================================
# 3. 创建自定义 Transformer
# ============================================================

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


# ============================================================
# 4. 获取 Spark 输出
# ============================================================

def get_spark_outputs(pipeline_model, dataframe):

    result = pipeline_model.transform(dataframe)

    outputs = [
        row["genre_indices"]
        for row in result.select("genre_indices").collect()
    ]

    return np.asarray(outputs, dtype=np.int64)


# ============================================================
# 5. 获取 Keras 输出
# ============================================================

def get_keras_outputs(model):

    inputs = tf.constant(
        [
            ["Action", "Comedy"],
            ["Drama", "Action"],
            ["Comedy", "Drama"],
            ["Unknown", "Action"],
        ],
        dtype=tf.string,
    )

    result = model(
        {"genres": inputs},
        training=False,
    )

    if isinstance(result, dict):
        tensor = result["genre_indices"]
    else:
        tensor = result

    return np.asarray(tensor.numpy(), dtype=np.int64)


# ============================================================
# 6. 主测试流程
# ============================================================

def main():

    spark = create_spark_session()

    try:

        # 清理之前的测试结果，避免覆盖冲突
        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR)

        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        dataframe = create_test_dataframe(spark)

        print("\n========== 1. FIT PIPELINE ==========")

        pipeline = KamaeSparkPipeline(
            stages=[create_transformer()]
        )

        fitted_pipeline = pipeline.fit(dataframe)

        original_spark_outputs = get_spark_outputs(
            fitted_pipeline,
            dataframe,
        )

        print("Original Spark outputs:")
        print(original_spark_outputs)

        # ====================================================
        # 7. 保存并加载 Spark PipelineModel
        # ====================================================

        print("\n========== 2. SAVE SPARK PIPELINE MODEL ==========")

        fitted_pipeline.write().save(str(PIPELINE_PATH))

        print("PipelineModel saved to:")
        print(PIPELINE_PATH)

        print("\n========== 3. LOAD SPARK PIPELINE MODEL ==========")

        loaded_pipeline = KamaeSparkPipelineModel.read().load(
            str(PIPELINE_PATH)
        )

        print("Loaded pipeline stages:")
        for stage in loaded_pipeline.stages:
            print(type(stage).__name__)
            print("Vocabulary:", stage.getLabelsArray())
            print("Input column:", stage.getInputCol())
            print("Output column:", stage.getOutputCol())

        loaded_spark_outputs = get_spark_outputs(
            loaded_pipeline,
            dataframe,
        )

        print("\nLoaded Spark outputs:")
        print(loaded_spark_outputs)

        np.testing.assert_array_equal(
            original_spark_outputs,
            loaded_spark_outputs,
        )

        print("Spark PipelineModel serialization test passed!")

        # ====================================================
        # 8. 构建原始 Keras 模型
        # ====================================================

        print("\n========== 4. BUILD KERAS MODEL ==========")

        input_schema = [
            {
                "name": "genres",
                "dtype": "string",
                "shape": (2,),
            }
        ]

        original_keras_model = fitted_pipeline.build_keras_model(
            input_schema=input_schema
        )

        original_keras_outputs = get_keras_outputs(
            original_keras_model
        )

        print("Original Keras outputs:")
        print(original_keras_outputs)

        # ====================================================
        # 9. 保存 Keras 模型
        # ====================================================

        print("\n========== 5. SAVE KERAS MODEL ==========")

        original_keras_model.save(str(KERAS_MODEL_PATH))

        print("Keras model saved to:")
        print(KERAS_MODEL_PATH)

        # ====================================================
        # 10. 加载 Keras 模型
        # ====================================================

        print("\n========== 6. LOAD KERAS MODEL ==========")

        loaded_keras_model = tf.keras.models.load_model(
            str(KERAS_MODEL_PATH)
        )

        print("Keras model loaded successfully.")

        loaded_keras_outputs = get_keras_outputs(
            loaded_keras_model
        )

        print("\nLoaded Keras outputs:")
        print(loaded_keras_outputs)

        np.testing.assert_array_equal(
            original_keras_outputs,
            loaded_keras_outputs,
        )

        print("Keras model serialization test passed!")

        # ====================================================
        # 11. 最终 Spark / Keras 一致性检查
        # ====================================================

        print("\n========== 7. FINAL PARITY CHECK ==========")

        expected = np.asarray(
            [
                [1, 2],
                [3, 1],
                [2, 3],
                [0, 1],
            ],
            dtype=np.int64,
        )

        print("Expected:")
        print(expected)

        print("Loaded Spark:")
        print(loaded_spark_outputs)

        print("Loaded Keras:")
        print(loaded_keras_outputs)

        np.testing.assert_array_equal(
            loaded_spark_outputs,
            expected,
        )

        np.testing.assert_array_equal(
            loaded_keras_outputs,
            expected,
        )

        np.testing.assert_array_equal(
            loaded_spark_outputs,
            loaded_keras_outputs,
        )

        print("\nAll serialization and parity tests passed!")

        print("\nGenerated files:")
        print("Spark PipelineModel:", PIPELINE_PATH)
        print("Keras model:", KERAS_MODEL_PATH)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
