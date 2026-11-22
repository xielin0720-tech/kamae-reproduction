
# Copyright [2024] Expedia, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import pyspark.sql.functions as F
import tensorflow as tf
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DataType,
    IntegerType,
    StringType,
)

from kamae.keras.core.backend import TENSORFLOW_ONLY
from kamae.keras.tensorflow.layers import StringIndexLayer
from kamae.spark.params import (
    SingleInputSingleOutputParams,
    StringIndexParams,
)
from kamae.spark.utils import indexer_udf

from .base import BaseTransformer


class ArrayStringIndexTransformer(
    BaseTransformer,
    StringIndexParams,
    SingleInputSingleOutputParams,
):
    """
    Transformer that applies a shared string vocabulary to every element
    of an array of strings.

    Spark input:
        ArrayType(StringType())

    Spark output:
        ArrayType(IntegerType())

    The Keras implementation reuses StringIndexLayer, which applies
    StringLookup elementwise to the input string tensor.
    """

    supported_backends = TENSORFLOW_ONLY
    jit_compatible = False

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        labelsArray: Optional[List[str]] = None,
        stringOrderType: Optional[str] = None,
        maskToken: Optional[str] = None,
        numOOVIndices: int = 1,
    ) -> None:
        super().__init__()

        self._setDefault(
            stringOrderType="frequencyDesc",
            numOOVIndices=1,
            maskToken=None,
        )

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        # ArrayType equality can also depend on containsNull metadata.
        # Validate the array element type explicitly in _transform instead.
        return None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        input_datatype = self.get_column_datatype(
            dataset=dataset,
            column_name=input_col,
        )

        if not isinstance(input_datatype, ArrayType):
            raise TypeError(
                f"Input column '{input_col}' must be an array, "
                f"but got {input_datatype}."
            )

        if not isinstance(input_datatype.elementType, StringType):
            raise TypeError(
                f"Elements of input column '{input_col}' must be strings, "
                f"but got {input_datatype.elementType}."
            )

        labels = self.getLabelsArray()

        if labels is None:
            raise ValueError(
                "labelsArray is not set. "
                "Provide a vocabulary before transforming the dataset."
            )

        num_oov_indices = self.getNumOOVIndices()
        mask_token = self.getMaskToken()

        def encode_array(values):
            if values is None:
                return None

            return [
                indexer_udf(
                    label=value,
                    labels=labels,
                    num_oov_indices=num_oov_indices,
                    mask_token=mask_token,
                )
                for value in values
            ]

        array_index_udf = F.udf(
            encode_array,
            ArrayType(
                IntegerType(),
                containsNull=False,
            ),
        )

        return dataset.withColumn(
            output_col,
            array_index_udf(F.col(input_col)),
        )

    def get_keras_layer(self) -> tf.keras.layers.Layer:
        return StringIndexLayer(
            name=self.getLayerName(),
            input_dtype=self.getInputKerasDtype(),
            output_dtype=self.getOutputKerasDtype(),
            vocabulary=self.getLabelsArray(),
            mask_token=self.getMaskToken(),
            num_oov_indices=self.getNumOOVIndices(),
        )
