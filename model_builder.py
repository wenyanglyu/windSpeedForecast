import tensorflow as tf
from keras import layers, models
import numpy as np


# Positional Encoding
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000,
                                                               (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads, dtype=tf.float32)


class TemporalEmbedding(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        position = tf.range(tf.shape(x)[1], dtype=tf.float32)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) *
                          -(tf.math.log(10000.0) / self.d_model))
        pe = tf.expand_dims(position, 1) * div_term
        pe = tf.concat([tf.sin(pe), tf.cos(pe)], axis=1)
        return x + tf.cast(pe, x.dtype)


# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# Multi-Head Attention Layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)


# Cross-Attention Layer
class CrossAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, context):
        return self.layernorm(x + self.mha(x, context, context, None))


# Transformer Encoder Layer
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='gelu'),  # Changed to GELU activation
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual connection


def power_loss(y_true, y_pred):
    """
    Custom loss focusing on cubic relationship of wind power (P ∝ V³)
    """
    # Extract wind speed components
    wind_true = y_true[..., 0]  # Assuming wind speed is first component
    wind_pred = y_pred[..., 0]

    # Calculate power (proportional to V³)
    power_true = tf.pow(wind_true, 3)
    power_pred = tf.pow(wind_pred, 3)

    return tf.reduce_mean(tf.abs((power_pred - power_true) / (power_true + 1e-6)))


def combined_loss(y_true, y_pred, alpha=1, beta=0, gamma=0):
    """
    Combined loss with rebalanced weights
    gamma: heavily reduced weight for power loss
    """
    power_cubic_loss = power_loss(y_true, y_pred)
    mae_loss = tf.keras.losses.MAE(y_true, y_pred)
    gradients = y_pred[:, 1:] - y_pred[:, :-1]
    gradient_loss = tf.reduce_mean(tf.abs(gradients))
    print(f"[Loss] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    return alpha * mae_loss + beta * gradient_loss + gamma * power_cubic_loss


def create_transformer(input_shape, d_model, num_heads, dff, num_layers, target_shape, rate=0.1):
    """
    Create an advanced transformer model for wind speed forecasting

    Args:
        input_shape: Tuple of input shape (7200,)
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dff: Dimension of feed forward network
        num_layers: Number of transformer layers
        target_shape: Shape of target output (144, 1)
        rate: Dropout rate
    """
    inputs = layers.Input(shape=input_shape)

    # Reshape input to 3D tensor
    seq_length = 240  # 7200 / 30 = 240
    features = 30  # 201 * 30 = 6030

    # Initial dense layer and reshaping
    x = layers.Dense(seq_length * features)(inputs)
    x = layers.Reshape((seq_length, features))(x)

    # Project to d_model dimensions
    x = layers.Dense(d_model)(x)

    # Add temporal embedding
    x = TemporalEmbedding(d_model)(x)

    # Transformer blocks with residual connections
    for _ in range(num_layers):
        # Self attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            value_dim=d_model // num_heads,
            dropout=rate
        )(x, x, x)

        # Add & Norm with residual
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        x = layers.Dropout(rate)(x)

        # Feed Forward with GELU activation
        ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dense(d_model)
        ])

        ffn_output = ffn(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        x = layers.Dropout(rate)(x)

    # Output processing
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers with residual connections
    x1 = layers.Dense(512, activation='gelu')(x)
    x1 = layers.Dropout(rate)(x1)
    x = layers.Add()([x, layers.Dense(x.shape[-1])(x1)])  # Residual connection

    x2 = layers.Dense(256, activation='gelu')(x)
    x2 = layers.Dropout(rate)(x2)
    x = layers.Add()([x, layers.Dense(x.shape[-1])(x2)])  # Residual connection

    x = layers.Dense(target_shape[0])(x)
    outputs = layers.Reshape(target_shape)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=combined_loss)

    return model


def save_model(model, filepath):
    """Save the model to a file"""
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load the model from a file"""
    print(f"Loading model from: {filepath}")

    # Import custom metrics from training.py
    from training import (
        real_speed_mae,
        real_speed_mse,
        power_mae,
        power_rmse
    )

    class TemporalEmbedding(layers.Layer):
        def __init__(self, d_model, trainable=True, dtype=None, **kwargs):
            super().__init__(trainable=trainable, dtype=dtype, **kwargs)
            self.d_model = d_model

        def get_config(self):
            config = super().get_config()
            config.update({
                "d_model": self.d_model,
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def call(self, x):
            position = tf.range(tf.shape(x)[1], dtype=tf.float32)
            div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) *
                              -(tf.math.log(10000.0) / self.d_model))
            pe = tf.expand_dims(position, 1) * div_term
            pe = tf.concat([tf.sin(pe), tf.cos(pe)], axis=1)
            return x + tf.cast(pe, x.dtype)

    custom_objects = {
        'combined_loss': combined_loss,
        'power_loss': power_loss,
        'TemporalEmbedding': TemporalEmbedding,
        'CrossAttention': CrossAttention,
        'TransformerEncoderLayer': TransformerEncoderLayer,
        'real_speed_mae': real_speed_mae,
        'real_speed_mse': real_speed_mse,
        'power_mae': power_mae,
        'power_rmse': power_rmse
    }

    try:
        model = tf.keras.models.load_model(
            filepath,
            custom_objects=custom_objects
        )
        print("✅ Model loaded successfully")

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise
