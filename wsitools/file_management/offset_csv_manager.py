import os


# Manage the csv file in which save the offset of two WSIs
class OffsetCSVManager:
    def __init__(self, offset_table_csv_fn):
        self.offset_csv = offset_table_csv_fn
        if self.offset_csv is None:
            self.offset_csv = "./example/wsi_pair_offset.csv"
        if not os.path.exists(offset_table_csv_fn):
            raise Exception("Offset file does not exist.")
        auto_offset_dict = {}
        gt_offset_dict = {}
        self.lines = open(offset_table_csv_fn, 'r').readlines()
        for l in self.lines[1:]:  # skip the first line
            if l.strip():
                ele = l.split(",")
                auto_offset_dict[ele[0]] = (ele[1], ele[2], ele[3])
                gt_offset_dict[ele[0]] = (ele[1], ele[4], ele[5])
        self.auto_offset_dict = auto_offset_dict
        self.gt_offset_dict = gt_offset_dict

    # lookup the offset from a table
    def lookup_table(self, fixed_uuid, float_uuid):
        # 0: None of them exist, return is (0, 0); 1: ground truth exist, auto_reg not, return ground truth
        # 2: auto_reg exist, ground truth not exist, return automatic result; 3: both of them exist, return ground truth
        if fixed_uuid is not None:   # lookup the offset from fixed wsi uuid
            float_wsi_uuid_validate = self.auto_offset_dict[fixed_uuid][0]
            if not float_uuid == float_wsi_uuid_validate:
                raise Exception("Float wsi uuid may be incorrect")
            auto_x = float(self.auto_offset_dict[fixed_uuid][1])
            auto_y = float(self.auto_offset_dict[fixed_uuid][2])
            gt_x = float(self.gt_offset_dict[fixed_uuid][1])
            gt_y = float(self.gt_offset_dict[fixed_uuid][2])
            if bool(auto_x) and bool(auto_y) and bool(gt_x) and bool(gt_y):
                state_indicator = 3
                offset = (gt_x, gt_y)
            elif not(bool(auto_x) and bool(auto_y) and bool(gt_x) and bool(gt_y)):
                state_indicator = 0
                offset = (0, 0)
            elif not(bool(auto_x) and bool(auto_y)) and bool(gt_x) and bool(gt_y):
                state_indicator = 1
                offset = (0, 0)
            elif bool(auto_x) and bool(auto_y) and not(bool(gt_x) and bool(gt_y)):
                state_indicator = 2
                offset = (0, 0)
            else:
                raise Exception("Incomplete Offset in the table")
        else:
            raise Exception("Need to specify the fixed image uuid")
        return offset, state_indicator

    def update_ground_truth(self, fixed_wsi_uuid, float_wsi_uuid, offset):
        updated_lines = self.lines
        fp = open(self.offset_csv, 'w')
        for idx, l in enumerate(self.lines[1:]):  # skip the first line
            if l.strip():
                ele = l.split(",")
                if ele[0] == fixed_wsi_uuid:
                    if ele[1] == float_wsi_uuid:
                        print("Fixed WSI uuid not in the file, append this case")
                        updated_lines[idx+1] = fixed_wsi_uuid + "," + float_wsi_uuid + "," + str(offset[0]) + "," + str(offset[1]) + "," + ele[4] + "," + ele[5] + "\n"
                        fp.writelines(updated_lines)
                        fp.close()
                        return True
                    else:
                        print("Fixed and float WSI uuid may not match, please check")
                        fp.close()
                        return False
        updated_lines.append(fixed_wsi_uuid + "," + float_wsi_uuid + "," + str(offset[0]) + "," + str(offset[1]) + "," + "," + "\n")
        fp.writelines(updated_lines)
        fp.close()
        return True

    def update_auto_registration(self, fixed_wsi_uuid, float_wsi_uuid, offset):
        updated_lines = self.lines
        fp = open(self.offset_csv, 'w')
        for idx, l in enumerate(self.lines[1:]):  # skip the first line
            if l.strip():
                ele = l.split(",")
                if ele[0] == fixed_wsi_uuid:
                    if ele[1] == float_wsi_uuid:
                        print("Fixed WSI uuid not in the file, append this case")
                        updated_lines[idx + 1] = fixed_wsi_uuid + "," + float_wsi_uuid + "," + ele[2] + "," + ele[3] + "," + str(offset[0]) + "," + str(offset[1]) + "\n"
                        fp.writelines(updated_lines)
                        fp.close()
                        return True
                    else:
                        print("Fixed and float WSI uuid may not match, please check")
                        fp.close()
                        return False
        updated_lines.append(fixed_wsi_uuid + "," + float_wsi_uuid + "," + "," + "," + str(offset[0]) + "," + str(offset[1]) + "\n")
        fp.writelines(updated_lines)
        fp.close()
        return True


# example
if __name__ == '__main__':
    print("see wsitools/examples/wsi_aligment.py to take examples")
