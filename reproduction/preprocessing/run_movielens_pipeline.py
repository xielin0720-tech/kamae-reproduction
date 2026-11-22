
import json
import time
from pathlib import Path

import numpy as np
import pyspark
import tensorflow as tf
import kamae

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    LongType,
    StringType,
)

from kamae.spark.pipeline import KamaeSparkPipeline
from kamae.spark.estimators import (
    StringIndexEstimator,
    OneHotEncodeEstimator,
)
from kamae.spark.transformers import HashIndexTransformer
from kamae.spark.transformers.string_to_string_list import (
    StringToStringListTransformer,
)
from kamae.spark.transformers.array_string_index import (
    ArrayStringIndexTransformer,
)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "reproduction" / "data" / "ml-1m"
RESULT_DIR = PROJECT_ROOT / "reproduction" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = RESULT_DIR / "spark_output.parquet"
KERAS_MODEL_PATH = RESULT_DIR / "preprocessing_model.keras"
VOCAB_PATH = RESULT_DIR / "genres_vocabulary.json"
LOG_PATH = RESULT_DIR / "experiment_log.md"


# ==========================================================
# SPARK SESSION
# ==========================================================

def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-MovieLens-1M-Reproduction")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


# ==========================================================
# LOAD MOVIELENS DATA
# ==========================================================

def load_movielens_data(spark):

    ratings_schema = StructType([
        StructField("UserID", IntegerType(), False),
        StructField("MovieID", IntegerType(), False),
        StructField("Rating", IntegerType(), False),
        StructField("Timestamp", LongType(), False),
    ])

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

    ratings = (
        spark.read
        .option("sep", "::")
        .schema(ratings_schema)
        .csv(str(DATA_DIR / "ratings.dat"))
    )

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

    # Join ratings, users and movies.
    joined = (
        ratings
        .join(users, on="UserID", how="inner")
        .join(movies, on="MovieID", how="inner")
    )

    # Convert categorical identifiers to strings.
    joined = (
        joined
        .withColumn("UserID", F.col("UserID").cast("string"))
        .withColumn("MovieID", F.col("MovieID").cast("string"))
        .withColumn("Occupation", F.col("Occupation").cast("string"))
    )

    return joined, movies


# ==========================================================
# BUILD GLOBAL GENRE VOCABULARY
# ==========================================================

def build_genre_vocabulary(movies):

    rows = (
        movies
        .select(
            F.explode(
                F.split(F.col("Genres"), r"\|")
            ).alias("Genre")
        )
        .filter(F.col("Genre").isNotNull())
        .distinct()
        .orderBy(F.col("Genre").asc())
        .collect()
    )

    return [row["Genre"] for row in rows]


# ==========================================================
# BUILD KAMAE PIPELINE
# ==========================================================

def build_pipeline(vocabulary):

    # UserID: Hash Index
    user_id_indexer = HashIndexTransformer(
        inputCol="UserID",
        outputCol="UserID_indexed",
        inputDtype="string",
        outputDtype="int",
        layerName="user_id_hash",
        numBins=10000,
    )

    # MovieID: String Index
    movie_id_indexer = StringIndexEstimator(
        inputCol="MovieID",
        outputCol="MovieID_indexed",
        inputDtype="string",
        outputDtype="int",
        layerName="movie_id_string_index",
        stringOrderType="alphabeticalAsc",
        numOOVIndices=1,
    )

    # Occupation: One-Hot Encoding
    occupation_encoder = OneHotEncodeEstimator(
        inputCol="Occupation",
        outputCol="Occupation_indexed",
        inputDtype="string",
        outputDtype="float",
        layerName="occupation_one_hot",
        stringOrderType="alphabeticalAsc",
        numOOVIndices=1,
    )

    # Genres: split into a fixed-length string array.
    genres_splitter = StringToStringListTransformer(
        inputCol="Genres",
        outputCol="GenresArray",
        inputDtype="string",
        outputDtype="string",
        layerName="genres_split",
        separator="|",
        listLength=6,
        defaultValue="",
    )

    # GenresArray: global string indexing.
    genres_indexer = ArrayStringIndexTransformer(
        inputCol="GenresArray",
        outputCol="GenreIndices",
        inputDtype="string",
        outputDtype="bigint",
        layerName="genres_string_index",
        labelsArray=vocabulary,
        stringOrderType="alphabeticalAsc",
        numOOVIndices=1,
    )

    return KamaeSparkPipeline(
        stages=[
            user_id_indexer,
            movie_id_indexer,
            occupation_encoder,
            genres_splitter,
            genres_indexer,
        ]
    )


# ==========================================================
# SPARK / KERAS PARITY CHECK
# ==========================================================

def check_parity(fitted_pipeline, spark_result):

    input_schema = [
        {"name": "UserID", "dtype": "string", "shape": ()},
        {"name": "MovieID", "dtype": "string", "shape": ()},
        {"name": "Occupation", "dtype": "string", "shape": ()},
        {"name": "Genres", "dtype": "string", "shape": ()},
    ]

    keras_model = fitted_pipeline.build_keras_model(
        input_schema=input_schema
    )

    # Use a deterministic sample for parity verification.
    sample_rows = (
        spark_result
        .orderBy("UserID", "MovieID", "Timestamp")
        .limit(256)
        .collect()
    )

    keras_inputs = {
        "UserID": np.asarray(
            [row["UserID"] for row in sample_rows],
            dtype=str,
        ),
        "MovieID": np.asarray(
            [row["MovieID"] for row in sample_rows],
            dtype=str,
        ),
        "Occupation": np.asarray(
            [row["Occupation"] for row in sample_rows],
            dtype=str,
        ),
        "Genres": np.asarray(
            [row["Genres"] for row in sample_rows],
            dtype=str,
        ),
    }

    keras_outputs = keras_model(
        keras_inputs,
        training=False,
    )

    output_columns = [
        "UserID_indexed",
        "MovieID_indexed",
        "Occupation_indexed",
        "GenreIndices",
    ]

    print("\nKeras output keys:", list(keras_outputs.keys()))

    for column in output_columns:

        spark_values = np.asarray(
            [row[column] for row in sample_rows]
        )

        keras_values = keras_outputs[column].numpy()

        print(f"\nChecking {column}")
        print("Spark shape:", spark_values.shape)
        print("Keras shape:", keras_values.shape)

        if spark_values.dtype.kind == "f":
            np.testing.assert_allclose(
                spark_values,
                keras_values,
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            np.testing.assert_array_equal(
                spark_values,
                keras_values,
            )

        print(f"{column}: PASSED")

    return keras_model


# ==========================================================
# EXPERIMENT LOG
# ==========================================================

def write_experiment_log(
    movie_count,
    vocabulary,
    fit_seconds,
    transform_seconds,
    save_seconds,
):

    content = f"""
# MovieLens 1M Kamae Reproduction

## Environment

- Python: 3.10
- Kamae: {getattr(kamae, '__version__', '3.5.0')}
- PySpark: {pyspark.__version__}
- TensorFlow: {tf.__version__}

## Dataset

- Dataset: MovieLens 1M
- Joined records: {movie_count}
- Genre vocabulary size: {len(vocabulary)}
- Maximum genres per movie: 6

## Pipeline

| Input | Operation | Output |
|---|---|---|
| UserID | Hash Index, 10000 bins | UserID_indexed |
| MovieID | String Index | MovieID_indexed |
| Occupation | One-Hot Encoding | Occupation_indexed |
| Genres | String split and padding, length 6 | GenresArray |
| GenresArray | Array String Index | GenreIndices |

## Genre Vocabulary

```json
{json.dumps(vocabulary, ensure_ascii=False, indent=2)}
```

## Execution Time

- Pipeline fit: {fit_seconds:.3f} seconds
- Spark transform and Parquet save: {transform_seconds:.3f} seconds
- Keras model save: {save_seconds:.3f} seconds

## Output Artifacts

- spark_output.parquet
- preprocessing_model.keras
- genres_vocabulary.json

## Validation

Spark/Keras parity was checked on a deterministic sample of 256 records.
"""

    LOG_PATH.write_text(content, encoding="utf-8")


# ==========================================================
# MAIN
# ==========================================================

def main():

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:

        # 1. Load data.
        print("\n========== 1. LOAD MOVIELENS ==========")

        joined, movies = load_movielens_data(spark)

        movie_count = joined.count()

        print("Joined record count:", movie_count)

        # 2. Build vocabulary.
        print("\n========== 2. BUILD GENRE VOCABULARY ==========")

        vocabulary = build_genre_vocabulary(movies)

        print("Vocabulary size:", len(vocabulary))
        print("Vocabulary:", vocabulary)

        VOCAB_PATH.write_text(
            json.dumps(vocabulary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 3. Build pipeline.
        print("\n========== 3. BUILD PIPELINE ==========")

        pipeline = build_pipeline(vocabulary)

        for i, stage in enumerate(pipeline.getStages(), start=1):
            print(i, stage.__class__.__name__)

        # 4. Fit pipeline.
        print("\n========== 4. FIT PIPELINE ==========")

        start = time.perf_counter()

        fitted_pipeline = pipeline.fit(joined)

        fit_seconds = time.perf_counter() - start

        print("Fit time:", fit_seconds)

        # 5. Transform full dataset and save Parquet.
        print("\n========== 5. SPARK TRANSFORM ==========")

        start = time.perf_counter()

        spark_result = fitted_pipeline.transform(joined)

        spark_result.printSchema()

        spark_result.write.mode("overwrite").parquet(
            str(PARQUET_PATH)
        )

        transform_seconds = time.perf_counter() - start

        print("Transform and save time:", transform_seconds)
        print("Parquet saved to:", PARQUET_PATH)

        # 6. Check Spark/Keras parity.
        print("\n========== 6. SPARK / KERAS PARITY ==========")

        keras_model = check_parity(
            fitted_pipeline,
            spark_result,
        )

        # 7. Save Keras model.
        print("\n========== 7. SAVE KERAS MODEL ==========")

        start = time.perf_counter()

        keras_model.save(str(KERAS_MODEL_PATH))

        save_seconds = time.perf_counter() - start

        print("Keras model saved to:", KERAS_MODEL_PATH)
        print("Keras save time:", save_seconds)

        # 8. Write experiment log.
        write_experiment_log(
            movie_count=movie_count,
            vocabulary=vocabulary,
            fit_seconds=fit_seconds,
            transform_seconds=transform_seconds,
            save_seconds=save_seconds,
        )

        print("\n========== 8. FINAL SUMMARY ==========")
        print("Records:", movie_count)
        print("Genre categories:", len(vocabulary))
        print("Spark output:", PARQUET_PATH)
        print("Keras model:", KERAS_MODEL_PATH)
        print("Experiment log:", LOG_PATH)
        print("\nMovieLens pipeline execution completed!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
