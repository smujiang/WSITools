import logging
import os


class WSI_CaseManager:
    """
    This code manages a csv file or a list, in which image pairs are defined.
    Example file of the csv file can be found at: ../file_management/example/case_pairs.csv
    See wsitools/examples/wsi_aligment.py to have more clues
    """

    def __init__(self, case_inventory_file=None):
        if not bool(case_inventory_file):  # if it's empty, load the default one
            logging.debug("loading the default image pair table")
            case_inventory_file = "../file_management/example/case_pairs.csv"
        if type(case_inventory_file) is list:  # if it's already the matched pairs
            self.counterpart_uuid_table = case_inventory_file  # list ([fixed_fn, float_fn])
        else:
            matched_pairs = []
            try:
                lines = open(case_inventory_file).readlines()
                for l in lines[1:]:  # skip the first line
                    if l.strip():
                        ele = l.strip().split(",")
                        fixed_fn = ele[0]
                        float_fn = ele[1]
                        matched_pairs.append([fixed_fn, float_fn])
                self.counterpart_uuid_table = matched_pairs
            except FileNotFoundError:
                raise Exception("Something went wrong when open the file")

    @staticmethod
    def get_wsi_fn_info(wsi_fn):
        if not os.path.isabs(wsi_fn):
            wsi_fn = os.path.abspath(wsi_fn)
        root_dir, fn = os.path.split(wsi_fn)
        uuid, ext = os.path.splitext(fn)
        return root_dir, uuid, ext

    def get_wsi_counterpart_uuid(self, wsi_name):
        for p in self.counterpart_uuid_table:
            if p[0] in wsi_name:
                return p[1]
            if p[1] in wsi_name:
                return p[0]
        return None

    def get_counterpart_fn(self, wsi_name, counterpart_root_dir, ext=".tiff"):
        counterpart_case_uuid = self.get_wsi_counterpart_uuid(wsi_name)
        if counterpart_case_uuid:
            return os.path.join(counterpart_root_dir, counterpart_case_uuid + ext)
        else:
            raise Exception("Can't find counterpart")


if __name__ == '__main__':
    print("Type help(WSI_CaseManager) to get more information")
