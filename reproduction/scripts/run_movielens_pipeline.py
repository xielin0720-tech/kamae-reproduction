import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from kamae.spark.utils.indexer_utils import collect_labels_array

# ---------------------------------------------------------
# 1. Project paths and existing MovieLens data loader
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = ROOT_DIR / "reproduction" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "reproduction" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PREPROCESSING_DIR))

from movielens_preprocessing import (  # noqa: E402
    load_movielens_data,
    build_movielens_dataframe,
)

# ---------------------------------------------------------
# 2. Kamae imports
# ---------------------------------------------------------

from kamae.spark.pipeline import KamaeSparkPipeline  # noqa: E402

from kamae.spark.transformers import (  # noqa: E402
    HashIndexTransformer,
    StringToStringListTransformer,
    ArrayStringIndexTransformer,
)

from kamae.spark.estimators import (  # noqa: E402
    StringIndexEstimator,
    OneHotEncodeEstimator,
)


# ---------------------------------------------------------
# 3. Spark session
# ---------------------------------------------------------

def create_spark_session():
    return (
        SparkSession.builder
        .appName("Kamae-MovieLens-Full-Reproduction")
        .master("local[1]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


# ---------------------------------------------------------
# 4. Load MovieLens 1M
# ---------------------------------------------------------

def load_training_dataframe(spark):
    users, movies, ratings = load_movielens_data(spark)

    print("\n========== Raw MovieLens 1M counts ==========")
    print("Users:", users.count())
    print("Movies:", movies.count())
    print("Ratings:", ratings.count())

    train_ml = build_movielens_dataframe(
        users,
        movies,
        ratings,
    )

    print("\n========== Joined MovieLens dataframe ==========")
    print("Total rows:", train_ml.count())

    train_ml.printSchema()
    train_ml.show(5, truncate=False)

    return train_ml


# ---------------------------------------------------------
# 5. Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessing_pipeline(train_ml):

    # MovieLens 1M genres vocabulary.
    # The vocabulary was obtained from the dataset:
    # 18 distinct genres, in alphabetical order.
    # genres_vocabulary = [
    #     "Action",
    #     "Adventure",
    #     "Animation",
    #     "Children's",
    #     "Comedy",
    #     "Crime",
    #     "Documentary",
    #     "Drama",
    #     "Fantasy",
    #     "Film-Noir",
    #     "Horror",
    #     "Musical",
    #     "Mystery",
    #     "Romance",
    #     "Sci-Fi",
    #     "Thriller",
    #     "War",
    #     "Western",
    # ]

    # Create the same genre-splitting transformer used in the pipeline.

    genres_splitter = StringToStringListTransformer(
        inputCol="Genres",
        outputCol="Genres_split",
        inputDtype="string",
        outputDtype="string",
        separator="|",
        listLength=6,
        defaultValue="PADDED",
        layerName="genres_split_to_array_transform",
    )

    # Apply the genre splitter to the training dataframe.
    genres_split_df = genres_splitter.transform(train_ml)

    # Collect genre vocabulary using Kamae's native implementation.
    # PADDED is excluded and labels are ordered by descending frequency.
    genres_vocabulary = collect_labels_array(
        dataset=genres_split_df,
        column=F.col("Genres_split"),
        column_datatype=genres_split_df.schema["Genres_split"].dataType,
        string_order_type="frequencyDesc",
        mask_token="PADDED",
    )

    print("Genres vocabulary (frequencyDesc):")
    for index, genre in enumerate(genres_vocabulary):
        print(f"{index + 2}: {genre}")

    print("Vocabulary size:", len(genres_vocabulary))

    stages = [

        # Stage 1: Hash-index UserID.
        HashIndexTransformer(
            inputCol="UserID",
            outputCol="UserID_indexed",
            inputDtype="string",
            outputDtype="int",
            numBins=10000,
            layerName="user_hash_indexer",
        ),

        # Stage 2: Learn MovieID vocabulary from the data.
        StringIndexEstimator(
            inputCol="MovieID",
            outputCol="MovieID_indexed",
            inputDtype="string",
            outputDtype="int",
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
            layerName="movie_id_string_indexer",
        ),

        # Stage 3: One-hot encode Occupation.
        OneHotEncodeEstimator(
            inputCol="Occupation",
            outputCol="Occupation_indexed",
            inputDtype="string",
            outputDtype="float",
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
            dropUnseen=True,
            layerName="occupation_one_hot_encoder",
        ),

        # Stage 4: Split Genres into a fixed-length string array.
        genres_splitter,

        # Stage 5: Index every element in the Genres array.
        # This is our custom adapter because the standard
        # StringIndexTransformer handles scalar strings.
        ArrayStringIndexTransformer(
            inputCol="Genres_split",
            outputCol="Genres_indexed",
            inputDtype="string",
            outputDtype="int",
            labelsArray=genres_vocabulary,
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
            maskToken="PADDED",
            layerName="genres_string_indexer",
        ),
    ]

    return KamaeSparkPipeline(stages=stages)


# ---------------------------------------------------------
# 6. Run Spark pipeline and export Keras model
# ---------------------------------------------------------

def main():

    spark = create_spark_session()

    try:
        print("\n========== Step 1: Load MovieLens 1M ==========")

        train_ml = load_training_dataframe(spark)

        print("\n========== Step 2: Build Kamae pipeline ==========")

        pipeline = build_preprocessing_pipeline(train_ml)

        print("\n========== Step 3: Fit pipeline ==========")

        fitted_pipeline = pipeline.fit(train_ml)

        print("\n========== Step 4: Transform MovieLens data ==========")

        spark_output = fitted_pipeline.transform(train_ml)

        print("Transformed row count:", spark_output.count())

        print("\n========== Transformed schema ==========")
        spark_output.printSchema()

        print("\n========== Transformed samples ==========")

        spark_output.select(
            "UserID",
            "UserID_indexed",
            "MovieID",
            "MovieID_indexed",
            "Occupation",
            "Occupation_indexed",
            "Genres",
            "Genres_split",
            "Genres_indexed",
        ).show(10, truncate=False)

        # Save Spark preprocessing results.
        parquet_path = OUTPUT_DIR / "spark_output.parquet"

        print("\n========== Step 5: Save Spark output ==========")

        (
            spark_output.write
            .mode("overwrite")
            .parquet(str(parquet_path))
        )

        print("Spark output saved to:", parquet_path)

        # -------------------------------------------------
        # 7. Build Keras preprocessing model
        # -------------------------------------------------

        print("\n========== Step 6: Build Keras model ==========")

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

        keras_model = fitted_pipeline.build_keras_model(
            input_schema=input_schema,
        )

        keras_model.summary()

        # Save Keras model.
        keras_path = OUTPUT_DIR / "preprocessing_model.keras"

        print("\n========== Step 7: Save Keras model ==========")

        keras_model.save(str(keras_path))

        print("Keras model saved to:", keras_path)

        print("\n========== Reproduction pipeline completed ==========")
        print("Spark output:", parquet_path)
        print("Keras model:", keras_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

