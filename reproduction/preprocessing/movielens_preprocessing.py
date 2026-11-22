
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
)


# MovieLens 1M 数据目录
DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ml-1m"
)


def create_spark_session():
    """创建 SparkSession。"""
    return (
        SparkSession.builder
        .appName("Kamae-MovieLens-Reproduction")
        .master("local[*]")
        .getOrCreate()
    )


def load_movielens_data(spark):
    """读取 MovieLens 1M 原始数据。"""

    users_schema = StructType([
        StructField("UserID", IntegerType(), False),
        StructField("Gender", StringType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Occupation", IntegerType(), True),
        StructField("Zip-code", StringType(), True),
    ])

    movies_schema = StructType([
        StructField("MovieID", IntegerType(), False),
        StructField("Title", StringType(), True),
        StructField("Genres", StringType(), True),
    ])

    ratings_schema = StructType([
        StructField("UserID", IntegerType(), False),
        StructField("MovieID", IntegerType(), False),
        StructField("Rating", IntegerType(), True),
        StructField("Timestamp", LongType(), True),
    ])

    users = (
        spark.read
        .option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .schema(users_schema)
        .csv(str(DATA_DIR / "users.dat"))
    )

    movies = (
        spark.read
        .option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .schema(movies_schema)
        .csv(str(DATA_DIR / "movies.dat"))
    )

    ratings = (
        spark.read
        .option("sep", "::")
        .schema(ratings_schema)
        .csv(str(DATA_DIR / "ratings.dat"))
    )

    return users, movies, ratings


def build_movielens_dataframe(users, movies, ratings):
    """关联用户、电影和评分数据。"""

    result = (
        ratings
        .join(users, on="UserID", how="inner")
        .join(movies, on="MovieID", how="inner")
        .withColumn(
            "Occupation",
            F.col("Occupation").cast(StringType())
        )
        .withColumn(
            "MovieID",
            F.col("MovieID").cast(StringType())
        )
        .withColumn(
            "UserID",
            F.col("UserID").cast(StringType())
        )
    )

    return result


if __name__ == "__main__":

    spark = create_spark_session()

    try:
        users, movies, ratings = load_movielens_data(spark)

        print("========== 原始数据数量 ==========")
        print("Users:", users.count())
        print("Movies:", movies.count())
        print("Ratings:", ratings.count())

        train_ml = build_movielens_dataframe(
            users, movies, ratings
        )

        print("========== 关联后的数据 ==========")
        print("Total:", train_ml.count())

        train_ml.printSchema()
        train_ml.show(5, truncate=False)

    finally:
        spark.stop()
