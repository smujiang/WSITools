from xlrd import open_workbook
import os


class WSI_CaseManager:
    def __init__(self, case_inventory_xls=None, sheet=6, cols=(6, 9), rows=(8, -1)):
        if not bool(case_inventory_xls):    # if it's empty, load the default one
            case_inventory_xls = "/projects/shart/digital_pathology/data/PenMarking/model/Flotte Slide Master Inventory - TF.xlsx"
        MELF_Sheet_idx = sheet  # index of the sheet
        Marked_UUID_idx = cols[0]  # index of column save marked WSI
        Clean_UUID_idx = cols[1]  # index of column save clean WSI
        Start_row_idx = rows[0]  # index of start row which contains uuid
        matched_pairs = []
        try:
            wb = open_workbook(case_inventory_xls, 'r')
            wb_sheet = wb.sheet_by_index(MELF_Sheet_idx)
            if rows[1] == -1:   # load all the cases in the table
                cases_n = wb_sheet.nrows
            else:
                cases_n = rows[1]
            for row_idx in range(Start_row_idx, cases_n):
                marked_uuid = wb_sheet.cell(row_idx, Marked_UUID_idx).value
                clean_uuid = wb_sheet.cell(row_idx, Clean_UUID_idx).value
                matched_pairs.append([marked_uuid, clean_uuid])
            self.counterpart_uuid_table = matched_pairs
        except FileNotFoundError:
            print("Something went wrong when open the file")

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
            return os.path.join(counterpart_root_dir, counterpart_case_uuid+ext)
        else:
            raise Exception("Can't find counterpart")


# example
if __name__ == '__main__':
    print("see auto_wsi_matcher.py for examples")

