import tensorflow as tf
from tensorflow.keras import layers

class ConformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_multiplier=4, conv_kernel_size=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ffn1 = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(d_model * ff_multiplier, activation="swish"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.ffn2 = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(d_model * ff_multiplier, activation="swish"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.norm_mha = layers.LayerNormalization()
        self.conv = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Conv1D(filters=2*d_model, kernel_size=1, padding="same", activation="swish"),
            layers.Conv1D(filters=2*d_model, kernel_size=conv_kernel_size, padding="same", groups=2*d_model, activation="swish"),
            layers.Conv1D(filters=d_model, kernel_size=1, padding="same", activation="swish"),
            layers.Dropout(dropout)
        ])
        self.norm_conv = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = inputs + 0.5 * self.ffn1(inputs, training=training)
        mha_out = self.mha(x, x, training=training)
        x = x + self.dropout(mha_out, training=training)
        x = self.norm_mha(x)
        conv_out = self.conv(x, training=training)
        x = x + self.dropout(conv_out, training=training)
        x = self.norm_conv(x)
        x = x + 0.5 * self.ffn2(x, training=training)
        return x
