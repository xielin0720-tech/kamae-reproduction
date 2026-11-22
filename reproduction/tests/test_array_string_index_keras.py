
import numpy as np
import tensorflow as tf

from kamae.keras.tensorflow.layers import StringIndexLayer


def main():
    vocabulary = [
        "Action",
        "Comedy",
        "Drama",
    ]

    layer = StringIndexLayer(
        name="genre_string_index",
        vocabulary=vocabulary,
        input_dtype="string",
        num_oov_indices=1,
        mask_token=None,
    )

    inputs = tf.constant(
        [
            ["Action", "Comedy"],
            ["Drama", "Action"],
            ["Fantasy", "Comedy"],
        ],
        dtype=tf.string,
    )

    outputs = layer(inputs)

    expected = np.array(
        [
            [1, 2],
            [3, 1],
            [0, 2],
        ],
        dtype=np.int64,
    )

    actual = outputs.numpy()

    print("========== Keras 输入 ==========")
    print(inputs.numpy())

    print("========== Keras 输出 ==========")
    print(actual)

    print("========== 结果校验 ==========")
    print("Actual:")
    print(actual)
    print("Expected:")
    print(expected)

    np.testing.assert_array_equal(actual, expected)

    print("Keras StringIndexLayer test passed!")


if __name__ == "__main__":
    main()
