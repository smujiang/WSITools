import os


# Manager class label text and ID. CSV file content like below:
# Label,ID
# lymph,0
# dermis,1
class ClassLabelCSVManager:
    def __init__(self, class_label_id_csv_fn):
        if not os.path.exists(class_label_id_csv_fn):
            raise Exception("File does not exist.")
        if class_label_id_csv_fn is None:
            class_label_id_csv_fn = "./example/case_label_id.csv"
        self.class_label_id_csv_fn = class_label_id_csv_fn
        self.label_text_id_dict = {}
        self.lines = open(class_label_id_csv_fn, 'r').readlines()
        for l in self.lines[1:]:  # skip the first line
            if l.strip():
                ele = l.split(",")
                self.label_text_id_dict[ele[0]] = [int(ele[1]), int(ele[2])]  # key: label_text, value: (label_ID, priority)

    def get_label_text(self, label_id):
        for key in self.label_text_id_dict.keys():
            if self.label_text_id_dict.get(key)[0] == label_id:
                return key
        raise Exception("Can't find the label ID")

    def get_label_id(self, label_text):
        return self.label_text_id_dict[label_text][0]

    # After you get the annotations from the QuPath,
    # you may need to modify the priority of each region to deal with overlapping of annotation
    def get_priority(self, label_text):
        return self.label_text_id_dict[label_text][1]

    def update_file(self):
        fp = open(self.class_label_id_csv_fn, "w")
        wrt_str = self.lines[0:]  # get the head line
        for k in self.label_text_id_dict.keys():
            wrt_str += k + "," + str(self.label_text_id_dict.get(k)[0]) + "," + str(self.label_text_id_dict.get(k)[1]) + "\n"
        fp.write(wrt_str)
        fp.close()

    def update_priority(self, label_txt_priority_dict):
        """
        update the priority in the csv tabel
        :param label_txt_priority_dict: a dictionary contain the label text(key) and priority of this label(value)
        example: label_txt_priority_dict = {"lymph": 2, "dermis": 1}
        :return:
        """
        for k in self.label_text_id_dict.keys():
            self.label_text_id_dict[k] = [self.label_text_id_dict.get(k)[0], label_txt_priority_dict.get(k)]
        self.update_file()

