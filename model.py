# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings1 = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, scope=0,zero_pad=True)
        self.embeddings2 = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, scope=1,zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding1
            enc1 = tf.nn.embedding_lookup(self.embeddings1, x) # (N, T1, d_model)
            enc1 *= self.hp.d_model**0.5 # scale
            enc1 += positional_encoding(enc1, self.hp.maxlen1)
            enc1 = tf.layers.dropout(enc1, self.hp.dropout_rate, training=training)

            # embedding2
            enc2 = tf.nn.embedding_lookup(self.embeddings1, x) # (N, T1, d_model)
            enc2 *= self.hp.d_model**0.5 # scale
            enc2 += positional_encoding(enc2, self.hp.maxlen1)
            enc2 = tf.layers.dropout(enc2, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc1, enc2 = multihead_attention(queries=(enc1, enc2),
                                              keys=(enc1, enc2),
                                              values=(enc2, enc1),
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc1 = ff(enc1, num_units=[self.hp.d_ff, self.hp.d_model])
                    enc2 = ff(enc2, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = (enc1, enc2)
        return memory, sents1, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding1
            dec1 = tf.nn.embedding_lookup(self.embeddings1, decoder_inputs)  # (N, T2, d_model)
            dec1 *= self.hp.d_model ** 0.5  # scale
            dec1 += positional_encoding(dec1, self.hp.maxlen2)
            dec1 = tf.layers.dropout(dec1, self.hp.dropout_rate, training=training)

            # embedding2
            dec2 = tf.nn.embedding_lookup(self.embeddings2, decoder_inputs)  # (N, T2, d_model)
            dec2 *= self.hp.d_model ** 0.5  # scale
            dec2 += positional_encoding(dec2, self.hp.maxlen2)
            dec2 = tf.layers.dropout(dec2, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec1, dec2 = multihead_attention(queries=(dec1, dec2),
                                              keys=(dec1, dec2),
                                              values=(dec2, dec1),
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec1, dec2 = multihead_attention(queries=(dec1, dec2),
                                              keys=memory,
                                              values=(memory[1], memory[0]),
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec1 = ff(dec1, num_units=[self.hp.d_ff, self.hp.d_model])
                    dec2 = ff(dec2, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(tf.concat([self.embeddings1, self.embeddings2], axis=-1)) # (2 * d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', tf.concat([dec1, dec2], axis=-1), weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

