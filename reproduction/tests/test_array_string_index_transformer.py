
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from kamae.spark.transformers.array_string_index import (
    ArrayStringIndexTransformer,
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-Array-String-Index-Transformer-Test")
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
        schema = StructType([
            StructField(
                "genres",
                ArrayType(StringType(), containsNull=True),
                True,
            )
        ])

        data = [
            (["Action", "Comedy"],),
            (["Drama", "Action"],),
            (["Fantasy"],),
            ([],),
            (None,),
        ]

        df = spark.createDataFrame(data, schema)

        vocabulary = [
            "Action",
            "Comedy",
            "Drama",
        ]

        transformer = ArrayStringIndexTransformer(
            inputCol="genres",
            outputCol="genre_indices",
            labelsArray=vocabulary,
            stringOrderType="alphabeticalAsc",
            numOOVIndices=1,
        )

        result = transformer.transform(df)

        print("========== 转换结果 ==========")
        result.show(truncate=False)

        actual = [
            row["genre_indices"]
            for row in result.select("genre_indices").collect()
        ]

        expected = [
            [1, 2],
            [3, 1],
            [0],
            [],
            None,
        ]

        print("========== 结果校验 ==========")
        print("Actual:  ", actual)
        print("Expected:", expected)

        assert actual == expected, (
            f"Array string index test failed.\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}"
        )

        print("ArrayStringIndexTransformer Spark test passed!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
