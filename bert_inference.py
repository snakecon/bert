# coding=utf-8
from __future__ import print_function

import collections

import tensorflow as tf
from tqdm import tqdm

from tokenization import FullTokenizer


BertModelMeta = collections.namedtuple("BertModelMeta", ["model_file", "vocab_file"])
BertInputPackage = collections.namedtuple("BertInputPackage", "query")


class BertInference(object):
    """
    The bert model.
    """
    def __init__(self, bert_meta):
        self.graph = self._load_graph(bert_meta.model_file)

        self.tokenizer = FullTokenizer(
            vocab_file=bert_meta.vocab_file,
            do_lower_case=True)
        self.max_seq_length = 128

        # Input.
        self.input_ids = self.graph.get_tensor_by_name('infer/input_ids:0')
        self.word_ids = self.graph.get_tensor_by_name('infer/input_mask:0')
        self.segment_ids = self.graph.get_tensor_by_name('infer/segment_ids:0')
        # Output.
        self.predictions = self.graph.get_tensor_by_name('infer/loss/Softmax:0')

        self.sess = tf.Session(graph=self.graph)

        self.inference(BertInputPackage(u'预热一下'))

    def inference(self, bert_input):
        """
        Call model.
        """
        input_ids, input_mask, segment_ids = self._convert_single_example(bert_input.query)
        preds_evaluated = self.sess.run(self.predictions, feed_dict={
            self.input_ids: [input_ids],
            self.word_ids: [input_mask],
            self.segment_ids: [segment_ids]
        })

        return preds_evaluated

    def _load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="infer",
                op_dict=None,
                producer_op_list=None
            )

        return graph

    def _convert_single_example(self, text_a):
        tokens_a = self.tokenizer.tokenize(text_a)

        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        return input_ids, input_mask, segment_ids


if __name__ == '__main__':
    inference = BertInference(BertModelMeta(
        'export/output_graph.pb',
        'chinese_L-12_H-768_A-12/vocab.txt')
    )

    for i in tqdm(range(2)):
        result = inference.inference(BertInputPackage(u'小型车研究了各项尺寸轴距跟A级车好很多'))
        print(result)
