import os
import random


class CaseListManager:
    def __init__(self, case_list_txt, ext='.tiff'):
        if case_list_txt is None:
            case_list_txt = "./example/case_list.txt"
        self.case_list = []
        self.case_uuid_list = []
        lines = open(case_list_txt, 'r').readlines()
        for l in lines:
            if l.strip():
                if os.path.splitext(l.strip())[1] == ext:
                    uuid = os.path.split(l.strip())[1][0:-(len(ext)+1)]  # file name without ext
                    self.case_list.append(l.strip())
                    self.case_uuid_list.append(uuid)

    @staticmethod
    def export_case_list_from_dir(wsi_dir, output_txt, wsi_ext='.tiff'):
        # export all the cases to a txt file
        file_list = os.listdir(wsi_dir)
        wrt_str = ""
        for f in file_list:
            if os.path.splitext(f)[1] == wsi_ext:
                wrt_str += os.path.join(wsi_dir, f) + "\n"
        wrt_str = wrt_str.strip()
        fp = open(output_txt, 'w')
        fp.write(wrt_str)
        fp.close()

    def get_case_full_path(self, uuid):
        return self.case_list[self.case_uuid_list.index(uuid)]

    def get_case_uuid(self, wsi_full_path):
        return self.case_uuid_list[self.case_list.index(wsi_full_path)]

    # get file names and save to a txt
    def get_fn_list_from_case_list(self, output_file, ext='.tiff'):
        fp = open(output_file, 'w')
        wrt_str = ""
        for l in self.case_uuid_list:
            wrt_str += l + ext + "\n"
        fp.writelines(wrt_str)
        fp.close()

    def random_chose(self):
        rd_n = random.randint(0, len(self.case_list))
        return self.case_list[rd_n]

    def get_case_num(self):
        return len(self.case_uuid_list)




