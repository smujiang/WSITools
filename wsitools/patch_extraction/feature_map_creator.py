import tensorflow as tf
import numpy as np


class FeatureMapCreator:
    def __init__(self, feature_map_tsv):
        self.feature_map = {}
        self.keys = []
        self.data_types = []
        self.eval_str = []
        lines = open(feature_map_tsv, 'r').readlines()
        for l in lines[1:]:  # skip the first line
            if l.strip():
                ele = l.strip().split(',')
                self.keys.append(ele[0])
                self.data_types.append(ele[1])
                self.eval_str.append(ele[2])

    def get_attr_str_eval(self, data_type_str, value):
        if eval(data_type_str) is int:
            return self.int64_feature(value)
        elif eval(data_type_str) is str:
            return self.bytes_feature(value)
        elif eval(data_type_str) is bytes:
            return self.bytes_feature(value)
        else:
            raise Exception("Unsupported data type")

    def update_feature_map_eval(self, values):
        for k in range(len(self.keys)):
            self.feature_map[self.keys[k]] = self.get_attr_str_eval(self.data_types[k], values[k])
        return self.feature_map

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def print(self):
        print(self.feature_map)


# if __name__ == "__main__":
    # fmc = FeatureMapCreator("./feature_maps/basic_fm_P.csv")
    # fmc = FeatureMapCreator("./feature_maps/basic_fm_PL.csv")
    # fmc = FeatureMapCreator("./feature_maps/basic_fm_PP.csv")
    # fmc = FeatureMapCreator("./feature_maps/basic_fm_PPL.csv")
    # fmc.print()
