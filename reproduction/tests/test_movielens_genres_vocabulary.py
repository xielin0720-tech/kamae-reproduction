
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "reproduction" / "data" / "ml-1m"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("MovieLens-Genres-Vocabulary")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


def load_movies(spark):
    schema = StructType([
        StructField("MovieID", IntegerType(), False),
        StructField("Title", StringType(), True),
        StructField("Genres", StringType(), True),
    ])

    return (
        spark.read
        .option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .schema(schema)
        .csv(str(DATA_DIR / "movies.dat"))
    )


def main():
    spark = create_spark_session()

    try:
        movies = load_movies(spark)

        print("\n========== 1. MOVIE DATA ==========")
        print("Movie count:", movies.count())

        print("\n========== 2. NULL / EMPTY GENRES ==========")

        invalid_count = movies.filter(
            F.col("Genres").isNull()
            | (F.trim(F.col("Genres")) == "")
        ).count()

        print("Null or empty Genres:", invalid_count)

        print("\n========== 3. SPLIT GENRES ==========")

        movies_with_genres = movies.withColumn(
            "GenresArray",
            F.split(F.col("Genres"), r"\|")
        )

        movies_with_genres.select(
            "MovieID", "Title", "Genres", "GenresArray"
        ).show(10, truncate=False)

        print("\n========== 4. GENRE COUNT DISTRIBUTION ==========")

        movies_with_genres.withColumn(
            "GenreCount",
            F.size(F.col("GenresArray"))
        ).groupBy(
            "GenreCount"
        ).count().orderBy(
            "GenreCount"
        ).show()

        print("\n========== 5. GLOBAL VOCABULARY ==========")

        exploded = movies_with_genres.select(
            F.explode("GenresArray").alias("Genre")
        )

        genre_counts = (
            exploded
            .groupBy("Genre")
            .count()
            .orderBy(F.col("Genre").asc())
        )

        genre_counts.show(100, truncate=False)

        vocabulary = [
            row["Genre"]
            for row in genre_counts.select("Genre").collect()
            if row["Genre"] is not None
        ]

        print("Vocabulary size:", len(vocabulary))
        print("Vocabulary:", vocabulary)

        print("\n========== 6. MAX GENRES PER MOVIE ==========")

        max_genres = movies_with_genres.select(
            F.max(F.size("GenresArray")).alias("max_genres")
        ).first()["max_genres"]

        print("Maximum genres per movie:", max_genres)

        print("\n========== 7. SUMMARY ==========")
        print("Total movies:", movies.count())
        print("Vocabulary size:", len(vocabulary))
        print("Maximum genres per movie:", max_genres)
        print("Vocabulary order: alphabetical ascending")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
