import torch
from torch import nn, Tensor
from typing import Optional
from math import sqrt, log


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension: int, max_length: int, division_base: float = 10_000):
        """
        :param int embedding_dimension: the dimensions of the token embedding
        :param int max_length: the maximum length of the sequence
        :param int division_base: the base of the exponential division term
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_length = max_length
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embedding_dimension, 2) * (-log(division_base) / embedding_dimension))
        positional_encodings = torch.zeros(max_length, embedding_dimension)
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        self.positional_encodings: Tensor
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Adds the positional encoding of x to x and returns
        :param Tensor x: Tensor of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :param int offset: and integer offset to position
        :return: Tensor of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :rtype: Tensor
        """
        if x.shape[-1] != self.embedding_dimension:
            raise ValueError("Embedding dimensions do not match")
        if x.shape[-2] > self.max_length:
            raise ValueError("Sequence length exceeds max length")
        return x + self.positional_encodings[x.size(-2) + offset]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension: int, head_count: int):
        """
        :param int input_dimension: The dimensions of the token embeddings.
        :param int head_count: The number of heads. (input_dimensions % head_count must be 0)
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.head_count = head_count
        if self.input_dimension % self.head_count != 0:
            raise ValueError(
                "Input dimension must be divisible by number of heads")
        self.head_dimension = self.input_dimension // self.head_count
        self.query_transform = nn.Linear(input_dimension, input_dimension)
        self.key_transform = nn.Linear(input_dimension, input_dimension)
        self.value_transform = nn.Linear(input_dimension, input_dimension)
        self.output_transform = nn.Linear(input_dimension, input_dimension)

        self.attention_scaler = 1 / sqrt(self.head_dimension)

    def split_into_heads(self, x: Tensor) -> Tensor:
        """
        Split input dimensions for different heads
        :param x: Tensor of dimensions (batch_size, sequence_length, input_dimensions) | (sequence_length, input_dimensions)
        :return: Tensor of dimensions (batch_size, head_count, sequence_length, head_dimensions) | (head_count, sequence_length, head_dimensions)
        """
        *start_dimensions, _ = x.size()
        return x.view(*start_dimensions, self.head_count, self.head_dimension).transpose(-3, -2)

    def check_input_dimensions(self, x) -> bool:
        """
        Check Input dimensions
        :param Any x: the input to be checked
        :return:
        """
        return (x.dim() == 3 and x.shape[2] == self.input_dimension) or (
            x.dim() == 2 and x.shape[1] == self.input_dimension)

    def scaled_dot_product_attention(self, query_heads: Tensor, key_heads: Tensor, value_heads: Tensor,
                                     mask: None | torch.BoolTensor) -> Tensor:
        """
        Perform the scaled dot product attention on queries, key, and value heads
        :param Tensor query_heads: The Query Heads. A Tensor of shape (batch_size, head_count, query_sequence_length, head_dimensions) | (head_count, sequence_length, head_dimensions)
        :param Tensor key_heads: The Key Heads. A Tensor of shape (batch_size, head_count, sequence_length, head_dimensions) | (head_count, sequence_length, head_dimensions)
        :param Tensor value_heads: The Value Heads. A Tensor of shape (batch_size, head_count, sequence_length, head_dimensions) | (head_count, sequence_length, head_dimensions)
        :param Tensor | None mask: The mask to be applied on the attention scores. A Tensor of shape  (batch_size, query_sequence_length, sequence_length) | (query_sequence_length, sequence_length)
        :return: Value Heads scaled by attention scores: A Tensor of shape (batch_size, head_count, query_sequence_length, head_dimensions) | (head_count, query_sequence_length, head_dimensions)
        :rtype: Tensor
        """
        attention_scores = torch.matmul(
            query_heads, key_heads.transpose(-2, -1)
        )
        scaled_attention_scores = self.attention_scaler * attention_scores
        if mask is not None:
            scaled_attention_scores = scaled_attention_scores.masked_fill(
                mask.unsqueeze(-3) == False, torch.finfo(scaled_attention_scores.dtype).min)
        attention_probabilities = torch.softmax(
            scaled_attention_scores, dim=-1)
        output = torch.matmul(attention_probabilities, value_heads)
        return output

    def combine_heads(self, x: Tensor) -> Tensor:
        """
        Combines outputs from all the multiple heads
        :param Tensor x: A Tensor of shape (batch_size, head_count, sequence_length, head_dimensions) | (head_count, sequence_length, head_dimensions)
        :return: A Tensor of shape (batch_size, sequence_length, input_dimensions) | (sequence_length, input_dimensions)
        :rtype: Tensor
        """
        x = x.transpose(-2, -3)
        *start_dimensions, _, _ = x.size()
        return x.contiguous().view(*start_dimensions, self.input_dimension)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[torch.BoolTensor] = None) -> Tensor:
        """
        Performs Multi-Headed Attention
        :param queries: a Tensor of shape (batch_size, query_sequence_length, input_dimensions) | (query_sequence_length, input_dimensions)
        :param keys: a Tensor of shape (batch_size, sequence_length, input_dimensions) | (sequence_length, input_dimensions)
        :param values: a Tensor of shape (batch_size, sequence_length, input_dimensions) | (sequence_length, input_dimensions)
        :param mask: a Tensor of shape (batch_size, query_sequence_length, sequence_length) | (query_sequence_length, sequence_length)
        :return: a Tensor of shape (batch_size, query_sequence_length, input_dimensions) | (query_sequence_length, input_dimensions)
        :rtype: Tensor
        """
        if not self.check_input_dimensions(queries):
            raise ValueError(
                f"wrong queries dimensions, dimensions can be (batch_size, sequence_length, {self.input_dimension=}) or (sequence_length, {self.input_dimension=})")
        if not self.check_input_dimensions(keys):
            raise ValueError(
                f"wrong keys dimensions, dimensions can be (batch_size, sequence_length, {self.input_dimension=}) or (sequence_length, {self.input_dimension=})")
        if not self.check_input_dimensions(values):
            raise ValueError(
                f"wrong values dimensions, dimensions can be (batch_size, sequence_length, {self.input_dimension=}) or (sequence_length, {self.input_dimension=})")

        if not (queries.dim() == keys.dim() == values.dim()):
            raise ValueError(
                "Mismatching shapes"
            )

        if queries.dim() == 3 and not (queries.shape[0] == keys.shape[0] == values.shape[0]):
            raise ValueError(
                "Mismatching batch sizes"
            )

        if not (keys.shape[-2] == values.shape[-2]):
            raise ValueError(
                "Mismatching sequence lengths"
            )

        query_sequence_length = queries.shape[-2]
        sequence_length = keys.shape[-2]
        if mask is not None and (query_sequence_length, sequence_length) != mask.shape and (
                mask.shape[0], query_sequence_length, sequence_length) != mask.shape:
            raise ValueError(
                f"Shape dimension must be (batch_size, {query_sequence_length=}, {sequence_length=}) | ({query_sequence_length=}, {sequence_length=})")

        transformed_queries = self.query_transform(queries)
        transformed_keys = self.key_transform(keys)
        transformed_values = self.value_transform(values)

        query_heads = self.split_into_heads(transformed_queries)
        key_heads = self.split_into_heads(transformed_keys)
        value_heads = self.split_into_heads(transformed_values)
        self_attention_heads = self.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, mask)
        self_attention = self.combine_heads(self_attention_heads)
        output = self.output_transform(self_attention)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension: int, head_count: int, feed_forward_hidden_dimensions: tuple[int, ...],
                 attention_dropout_rate: float, feed_forward_dropout_rate: float):
        """
        :param int input_dimension: the dimensions of input embedding
        :param int head_count: the number of heads, must divide input_dimension
        :param tuple[int, ...] feed_forward_hidden_dimensions: a list of hidden layer dimensions to use for feed forward
        :param float attention_dropout_rate: dropout rate for the attention module
        :param float feed_forward_dropout_rate: dropout rate for the feed_forward_module
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.head_count = head_count
        self.feed_forward_hidden_dimensions = feed_forward_hidden_dimensions
        self.self_attention_module = MultiHeadAttention(
            input_dimension, head_count)
        self.self_attention_dropout = nn.Dropout(attention_dropout_rate)
        self.feed_forward_module = nn.Sequential()
        for in_dimensions, out_dimensions in zip((input_dimension,) + feed_forward_hidden_dimensions[:-1],
                                                 feed_forward_hidden_dimensions):
            self.feed_forward_module.append(
                nn.Linear(in_dimensions, out_dimensions))
            self.feed_forward_module.append(
                nn.Dropout(feed_forward_dropout_rate))
            self.feed_forward_module.append(nn.PReLU())
        self.feed_forward_module.append(
            nn.Linear(feed_forward_hidden_dimensions[-1] if feed_forward_hidden_dimensions else input_dimension,
                      input_dimension))
        self.feed_forward_module.append(nn.Dropout(feed_forward_dropout_rate))
        self.attention_norm = nn.LayerNorm(input_dimension)
        self.feed_forward_norm = nn.LayerNorm(input_dimension)

    def forward(self, x: Tensor, mask: Optional[torch.BoolTensor] = None) -> Tensor:
        """
        :param x: the input to the encoder of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :param mask: an optional boolean tensor indicating whether to mask certain attention scores of shape (batch_size, sequence_length, sequence_length) | (sequence_length, sequence_length)
        :return: the encoder output of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :rtype: Tensor
        """
        if x.dim() not in [2, 3] or x.shape[-1] != self.input_dimension:
            raise ValueError("Invalid input shape")
        attention_output = self.self_attention_dropout(
            self.self_attention_module(x, x, x, mask))
        x = self.attention_norm(x + attention_output)
        feed_forward_output = self.feed_forward_module(x)
        x = self.feed_forward_norm(x + feed_forward_output)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dimension: int, head_count: int, feed_forward_hidden_dimensions: tuple[int, ...],
                 attention_dropout_rate: float, feed_forward_dropout_rate: float, layer_count: int):
        super().__init__()
        self.transformer_encoder_layers = nn.ModuleList([TransformerEncoderLayer(input_dimension, head_count,
                                                                                 feed_forward_hidden_dimensions,
                                                                                 attention_dropout_rate,
                                                                                 feed_forward_dropout_rate) for _ in
                                                         range(layer_count)])
        self.layer_count = layer_count

    def forward(self, x: Tensor, ignore_mask: Optional[torch.BoolTensor] = None,
                mask: Optional[torch.BoolTensor] = None):
        """
        :param x: the input to the encoder of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :param ignore_mask: an optional boolean tensor indicating which tokens in the sequence to ignore during attention of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :param mask: an optional boolean tensor indicating whether to mask certain attention scores of shape (batch_size, sequence_length, sequence_length) | (sequence_length, sequence_length)
        :return: the encoder output of shape (batch_size, sequence_length, embedding_dimensions) | (sequence_length, embedding_dimensions)
        :rtype: Tensor
        """
        if ignore_mask is not None and ignore_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"Invalid padding mask, expected shape {x.shape} got shape {ignore_mask.shape}")
        if mask is not None:
            if mask.dim() == 2 and mask.shape != (x.shape[-2], x.shape[-2]):
                raise ValueError(
                    f"Invalid mask, expected shape {(x.shape[-2], x.shape[-2])} got shape {mask.shape}")
            if mask.dim() == 3 and mask.shape != (x.shape[0], x.shape[-2], x.shape[-2]):
                raise ValueError(
                    f"Invalid mask, expected shape {(x.shape[0], x.shape[-2], x.shape[-2])} got shape {mask.shape}")
        if ignore_mask is not None:
            ignore_mask = ignore_mask.unsqueeze(-1) & ignore_mask.unsqueeze(
                -2)  # convert to sequence x sequence mask format
            mask = ignore_mask if mask is None else ignore_mask & mask
        for layer in self.transformer_encoder_layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dimension: int, self_attention_head_count: int, cross_attention_head_count: int,
                 feed_forward_hidden_dimensions: tuple[int, ...],
                 self_attention_dropout_rate: float, cross_attention_dropout_rate: float,
                 feed_forward_dropout_rate: float):
        """
        :param int input_dimension: the dimensions of input embedding
        :param int self_attention_head_count: the number of heads, must divide input_dimension
        :param int cross_attention_head_count: the number of heads, must divide input_dimension
        :param tuple[int, ...] feed_forward_hidden_dimensions: a list of hidden layer dimensions to use for feed forward
        :param float self_attention_dropout_rate: dropout rate for the self attention module
        :param float cross_attention_dropout_rate: dropout rate for the cross attention module
        :param float feed_forward_dropout_rate: dropout rate for the feed_forward_module
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.feed_forward_hidden_dimensions = feed_forward_hidden_dimensions
        self.self_attention_module = MultiHeadAttention(
            input_dimension, self_attention_head_count)
        self.self_attention_dropout = nn.Dropout(self_attention_dropout_rate)

        self.feed_forward_module = nn.Sequential()
        for in_dimensions, out_dimensions in zip((input_dimension,) + feed_forward_hidden_dimensions[:-1],
                                                 feed_forward_hidden_dimensions):
            self.feed_forward_module.append(
                nn.Linear(in_dimensions, out_dimensions))
            self.feed_forward_module.append(
                nn.Dropout(feed_forward_dropout_rate))
            self.feed_forward_module.append(nn.PReLU())
        self.feed_forward_module.append(
            nn.Linear(feed_forward_hidden_dimensions[-1] if feed_forward_hidden_dimensions else input_dimension,
                      input_dimension))
        self.feed_forward_module.append(nn.Dropout(feed_forward_dropout_rate))
        self.self_attention_norm = nn.LayerNorm(input_dimension)
        self.cross_attention_norm = nn.LayerNorm(input_dimension)
        self.feed_forward_norm = nn.LayerNorm(input_dimension)
        self.cross_attention_module = MultiHeadAttention(
            input_dimension, cross_attention_head_count)
        self.cross_attention_dropout = nn.Dropout(cross_attention_dropout_rate)

    def forward(self, source: Tensor, target: Tensor,
                self_attention_mask: Optional[torch.BoolTensor] = None,
                cross_attention_mask: Optional[torch.BoolTensor] = None
                ):
        # TODO shape check

        self_attention_output = self.self_attention_dropout(
            self.self_attention_module(target, target, target, self_attention_mask))
        target = self.self_attention_norm(target + self_attention_output)

        cross_attention_output = self.cross_attention_dropout(
            self.cross_attention_module(target, source, source, cross_attention_mask))
        target = self.cross_attention_norm(target + cross_attention_output)

        feed_forward_output = self.feed_forward_module(target)
        target = self.feed_forward_norm(target + feed_forward_output)
        return target


class TransformerDecoder(nn.Module):
    def __init__(self, input_dimension: int, self_attention_head_count: int, cross_attention_head_count: int,
                 feed_forward_hidden_dimensions: tuple[int, ...],
                 self_attention_dropout_rate: float, cross_attention_dropout_rate: float,
                 feed_forward_dropout_rate: float, layer_count: int):
        super().__init__()
        self.transformer_decoder_layers = nn.ModuleList([TransformerDecoderLayer(input_dimension,
                                                                                 self_attention_head_count,
                                                                                 cross_attention_head_count,
                                                                                 feed_forward_hidden_dimensions,
                                                                                 self_attention_dropout_rate,
                                                                                 cross_attention_dropout_rate,
                                                                                 feed_forward_dropout_rate) for _ in
                                                         range(layer_count)])
        self.layer_count = layer_count

    def forward(self, source: Tensor, target: Tensor, source_ignore_mask: Optional[torch.BoolTensor] = None,
                target_ignore_mask: Optional[torch.BoolTensor] = None,
                self_attention_mask: Optional[torch.BoolTensor] = None,
                cross_attention_mask: Optional[torch.BoolTensor] = None):
        # TODO Shape check

        causal_mask = (
            1 - torch.triu(torch.ones(target.shape[-2], target.shape[-2]), diagonal=1)).bool().to(target.device)

        if target_ignore_mask is not None:
            target_ignore_mask_crossed = target_ignore_mask.unsqueeze(
                -1) & target_ignore_mask.unsqueeze(-2)
            causal_mask = causal_mask & target_ignore_mask_crossed

        if self_attention_mask is not None:
            causal_mask = causal_mask & self_attention_mask

        if source_ignore_mask is not None:
            ignore_mask = torch.ones(target.shape).bool(
            ) if target_ignore_mask is None else target_ignore_mask
            ignore_mask = ignore_mask.unsqueeze(
                -1) & source_ignore_mask.unsqueeze(-2)
            cross_attention_mask = ignore_mask if cross_attention_mask is None else cross_attention_mask & ignore_mask
        elif target_ignore_mask is not None:
            ignore_mask = target_ignore_mask.unsqueeze(
                -1) & torch.ones(source.shape).bool().unsqueeze(-2)
            cross_attention_mask = ignore_mask if cross_attention_mask is None else cross_attention_mask & ignore_mask

        for layer in self.transformer_decoder_layers:
            target = layer(source, target, causal_mask, cross_attention_mask)

        return target


class Transformer(nn.Module):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int, embedding_dimensions: int,
                 max_length: int, encoder_head_count: int,
                 encoder_feed_forward_hidden_dimensions: tuple[int, ...], encoder_attention_dropout_rate: float,
                 encoder_feed_forward_dropout_rate: float, encoder_layer_count: int,
                 decoder_self_attention_head_count: int, decoder_cross_attention_head_count: int,
                 decoder_feed_forward_dimensions: tuple[int, ...], decoder_self_attention_dropout_rate: float,
                 decoder_cross_attention_dropout_rate: float, decoder_feed_forward_dropout_rate: float,
                 decoder_layer_count: int):
        super().__init__()
        self.embedding_module = nn.Embedding(
            source_vocabulary_size, embedding_dimensions)
        self.positional_embeddings = PositionalEncoding(
            embedding_dimensions, max_length)
        self.transformer_encoder = TransformerEncoder(embedding_dimensions, encoder_head_count,
                                                      encoder_feed_forward_hidden_dimensions,
                                                      encoder_attention_dropout_rate, encoder_feed_forward_dropout_rate,
                                                      encoder_layer_count)
        self.transformer_decoder = TransformerDecoder(embedding_dimensions, decoder_self_attention_head_count,
                                                      decoder_cross_attention_head_count,
                                                      decoder_feed_forward_dimensions,
                                                      decoder_self_attention_dropout_rate,
                                                      decoder_cross_attention_dropout_rate,
                                                      decoder_feed_forward_dropout_rate, decoder_layer_count)
        self.target_decoder = nn.Linear(
            embedding_dimensions, target_vocabulary_size)

    def forward(self, source: Tensor, target: Tensor, source_ignore_mask: Optional[torch.BoolTensor] = None,
                target_ignore_mask: Optional[torch.BoolTensor] = None, encoder_mask: Optional[torch.BoolTensor] = None,
                decoder_self_attention_mask: Optional[torch.BoolTensor] = None,
                decoder_cross_attention_mask: Optional[torch.BoolTensor] = None) -> Tensor:
        # TODO shape check documentation
        source, target = self.embedding_module(
            source), self.embedding_module(target)
        source, target = self.positional_embeddings(
            source), self.positional_embeddings(target, offset=source.shape[-2])
        source = self.transformer_encoder(
            source, source_ignore_mask, encoder_mask)
        target = self.transformer_decoder(source, target, source_ignore_mask, target_ignore_mask,
                                          decoder_self_attention_mask, decoder_cross_attention_mask)
        decoded_targets = self.target_decoder(target)
        return decoded_targets

    def beam_search(self, source: list[int], target_start: list[int], device: torch.device, beam_size: int,
                    end_token_index: int, max_depth=100, max_results: int = 5) -> list[tuple[list[int], float]]:
        hypotheses: list[tuple[list[int], float]] = [(target_start, 0.)]
        source: torch.Tensor = torch.tensor(source).to(device)
        source = self.embedding_module(source)
        source = self.positional_embeddings(source)
        source = self.transformer_encoder(source)
        depth = 0

        results: list[tuple[list[int], float]] = []
        while hypotheses and depth < max_depth and len(results) < max_results:
            target = [i[0] for i in hypotheses]

            target = torch.tensor(target).to(device)
            repeated_source = source.repeat(
                target.shape[0], *[1] * source.dim())
            target = self.positional_embeddings(self.embedding_module(target))
            predictions = self.target_decoder(
                self.transformer_decoder(repeated_source, target))[:, -1]
            predictions = torch.log_softmax(predictions, dim=-1)
            top_predictions = torch.topk(predictions, k=beam_size)
            top_predictions_indices = top_predictions.indices.tolist()
            top_predictions_log_probabilities = top_predictions.values.tolist()

            new_hypotheses = []
            for (prefix, log_probability), top_prediction, prediction_log_probabilities in zip(hypotheses,
                                                                                               top_predictions_indices,
                                                                                               top_predictions_log_probabilities):
                for prediction, prediction_log_probability in zip(top_prediction, prediction_log_probabilities):
                    if prediction == end_token_index:
                        results.append(
                            ([*prefix, end_token_index], log_probability + prediction_log_probability))
                    else:
                        new_hypotheses.append(
                            ([*prefix, prediction], log_probability + prediction_log_probability))

            new_hypotheses.sort(key=lambda n: n[-1], reverse=True)
            hypotheses = new_hypotheses[:beam_size]

            depth += 1

        results.extend(hypotheses)
        results.sort(key=lambda n: n[-1], reverse=True)

        return results
