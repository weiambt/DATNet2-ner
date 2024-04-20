def compute_tgt_logits(w_emb, c_emb):
    char_cnn = char_cnn_hw(c_emb, self.cfg.kernel_sizes, self.cfg.filters, self.cfg.char_dim, self.cfg.hw_layer,
                           activation=tf.tanh, name="char_cnn_hw")
    emb = tf.layers.dropout(tf.concat([w_emb, char_cnn], axis=-1), rate=self.cfg.emb_drop_rate,
                            training=self.training)
    rnn_feats = bi_rnn(emb, self.tgt_seq_len, self.training, self.cfg.tgt_num_units, self.cfg.rnn_drop_rate,
                       activation=tf.tanh, concat=self.cfg.concat_rnn, name="tgt_birnn")
    share_rnn_feats = bi_rnn(emb, self.tgt_seq_len, self.training, self.cfg.share_num_units,
                             self.cfg.rnn_drop_rate, activation=tf.tanh, concat=self.cfg.concat_rnn,
                             name="share_birnn")
    rnn_feats = tf.concat([rnn_feats, share_rnn_feats], axis=-1)
    logits = tf.layers.dense(rnn_feats, units=self.vocab.tgt_label_size, reuse=tf.AUTO_REUSE, name="tgt_proj")
    transition, crf_loss = crf_layer(logits, self.tgt_labels, self.tgt_seq_len, self.vocab.tgt_label_size,
                                     name="crf" if self.cfg.share_label else "tgt_crf")
    dis_loss = discriminator(rnn_feats, self.domain_labels, 2, self.cfg.grad_rev_rate, self.cfg.alpha,
                             self.cfg.gamma, self.cfg.disc, name="discriminator")
    if dis_loss is not None:
        crf_loss = crf_loss + dis_loss
    return logits, transition, crf_loss