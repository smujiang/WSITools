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
        self.label_text_id_dict = {}
        lines = open(class_label_id_csv_fn, 'r').readlines()
        for l in lines[1:]:  # skip the first line
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




