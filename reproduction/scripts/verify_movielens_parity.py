import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import kamae

from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("MovieLens-Spark-Keras-Parity")
    .master("local[1]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.default.parallelism", "1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:
    # 1. Load Spark output
    print("\n========== 1. Load Spark output ==========")

    spark_df = spark.read.parquet(
        r".\reproduction\outputs\spark_output.parquet"
    )

    sample_rows = (
        spark_df
        .select(
            "Genres",
            "MovieID",
            "Occupation",
            "UserID",
            "Genres_indexed",
            "MovieID_indexed",
            "Occupation_indexed",
            "UserID_indexed",
        )
        .limit(128)
        .collect()
    )

    if not sample_rows:
        raise RuntimeError("No rows found.")

    print("Sample count:", len(sample_rows))

    # 2. Load Keras model
    print("\n========== 2. Load Keras model ==========")

    model = tf.keras.models.load_model(
        r".\reproduction\outputs\preprocessing_model.keras"
    )

    print("Model loaded.")
    print("Inputs:", [x.name for x in model.inputs])
    print("Outputs:", model.output_names)

    # 3. Construct native TensorFlow string tensors
    print("\n========== 3. Prepare TensorFlow inputs ==========")

    def make_input(column):
        values = [
            str(row[column])
            for row in sample_rows
        ]

        # Shape: (batch, 1, 1)
        values = np.asarray(values, dtype=object).reshape(-1, 1, 1)

        return tf.convert_to_tensor(values, dtype=tf.string)

    keras_inputs = {
        "Genres": make_input("Genres"),
        "MovieID": make_input("MovieID"),
        "Occupation": make_input("Occupation"),
        "UserID": make_input("UserID"),
    }

    for name, value in keras_inputs.items():
        print(
            f"{name}: shape={value.shape}, dtype={value.dtype}"
        )

    # 4. Call model directly instead of model.predict()
    print("\n========== 4. Run Keras model ==========")

    keras_outputs = model(
        keras_inputs,
        training=False
    )

    # Kamae returns a dictionary keyed by transformed column names.
    if isinstance(keras_outputs, dict):
        print("Output dictionary keys:", list(keras_outputs.keys()))

        output_mapping = [
            "Genres_indexed",
            "MovieID_indexed",
            "Occupation_indexed",
            "UserID_indexed",
        ]

        missing = [
            name for name in output_mapping
            if name not in keras_outputs
        ]

        if missing:
            raise RuntimeError(
                f"Missing expected output keys: {missing}"
            )

        keras_outputs = [
            keras_outputs[name]
            for name in output_mapping
        ]

    elif isinstance(keras_outputs, (list, tuple)):
        keras_outputs = list(keras_outputs)

    else:
        keras_outputs = [keras_outputs]

    keras_outputs = [
        output.numpy() if hasattr(output, "numpy")
        else np.asarray(output)
        for output in keras_outputs
    ]

    print("Keras execution completed.")
    print("Number of outputs:", len(keras_outputs))

    # 5. Compare outputs
    print("\n========== 5. Spark / Keras parity comparison ==========")

    output_mapping = [
        "Genres_indexed",
        "MovieID_indexed",
        "Occupation_indexed",
        "UserID_indexed",
    ]

    all_passed = True

    for index, spark_column in enumerate(output_mapping):

        keras_result = np.asarray(keras_outputs[index])

        spark_values = [
            row[spark_column]
            for row in sample_rows
        ]

        expected = np.asarray(spark_values)

        # Convert Spark values to the same shape as Keras outputs.
        expected = expected.reshape(keras_result.shape)

        print(f"\n--- {spark_column} ---")
        print("Spark shape:", expected.shape)
        print("Keras shape:", keras_result.shape)

        if expected.shape != keras_result.shape:
            passed = False
            print("RESULT: FAIL - shape mismatch")

        elif np.issubdtype(expected.dtype, np.integer):
            passed = np.array_equal(expected, keras_result)
            print("RESULT:", "PASS" if passed else "FAIL")

        else:
            passed = np.allclose(
                expected.astype(np.float32),
                keras_result.astype(np.float32),
                rtol=1e-6,
                atol=1e-6,
            )
            print("RESULT:", "PASS" if passed else "FAIL")

        if not passed:
            all_passed = False

            if expected.shape == keras_result.shape:
                differences = np.argwhere(expected != keras_result)

                print("Mismatch count:", len(differences))

                if len(differences) > 0:
                    first = tuple(differences[0])

                    print("First mismatch:", first)
                    print("Spark value:", expected[first])
                    print("Keras value:", keras_result[first])

                    sample_index = first[0]
                    row = sample_rows[sample_index]

                    print("Original input:")
                    print({
                        "Genres": row["Genres"],
                        "MovieID": row["MovieID"],
                        "Occupation": row["Occupation"],
                        "UserID": row["UserID"],
                    })

    # 6. Final result
    print("\n========== Final result ==========")

    if all_passed:
        print("PASS: All four preprocessing outputs match.")
    else:
        print("FAIL: At least one preprocessing output differs.")

finally:
    spark.stop()
    print("\nSpark stopped.")
