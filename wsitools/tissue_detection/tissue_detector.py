import numpy as np
import os
from scipy import ndimage
from skimage.color import rgb2lab
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import joblib


class TissueDetector:
    def __init__(self, name, threshold=0.5, training_files=""):
        self.name = name
        self.threshold = threshold
        self.tsv_name = training_files

    def read_training_dim(self, feature_dim):
        tsv_cols = np.loadtxt(self.tsv_name, delimiter="\t", skiprows=1, usecols=range(feature_dim+1))
        return tsv_cols[:, 0:feature_dim + 1]

    def get_gnb_model(self):
        if not os.path.exists(self.tsv_name):
            return self.get_default_gnb_model()
        else:
            bkg_train_data = self.read_training_dim(3)
            gnb_bkg = GaussianNB()
            gnb_bkg.fit(bkg_train_data[:, 1:], bkg_train_data[:, 0])
        return gnb_bkg

    def get_svm_model(self):
        if not os.path.exists(self.tsv_name):
            return self.get_default_gnb_model()
        else:
            bkg_train_data = self.read_training_dim(3)
            svm_bkg = svm.SVC()
            svm_bkg.fit(bkg_train_data[:, 1:], bkg_train_data[:, 0])
        return svm_bkg

    def save_gnb_model(self, save_fn):
        gnb_classifier = self.get_gnb_model()
        joblib.dump(gnb_classifier, save_fn)

    @staticmethod
    def load_gnb_model(gnb_model_fn):
        gnb_model = joblib.load(gnb_model_fn)
        return gnb_model

    def get_default_gnb_model(self):
        cwd = os.path.dirname(__file__)
        self.tsv_name = os.path.join(cwd, 'model_files/HE_tissue_others.tsv')  # this file is created by our annotation tool
        bkg_train_data = self.read_training_dim(3)
        gnb_bkg = GaussianNB()
        gnb_bkg.fit(bkg_train_data[:, 1:], bkg_train_data[:, 0])
        return gnb_bkg

    def get_default_svm_model(self):
        cwd = os.path.dirname(__file__)
        self.tsv_name = os.path.join(cwd, 'model_files/HE_tissue_others.tsv')  # this file is created by our annotation tool
        bkg_train_data = self.read_training_dim(3)
        gnb_bkg = SVM()
        gnb_bkg.fit(bkg_train_data[:, 1:], bkg_train_data[:, 0])
        return gnb_bkg

    def predict(self, wsi_thumb_img, open_operation=False):
        if self.name == "LAB_Threshold":
            lab_img = rgb2lab(wsi_thumb_img)
            l_img = lab_img[:, :, 0]
            # tissue is darker than background, recommend threshold value: 85
            binary_img_array_1 = np.array(0 < l_img)
            binary_img_array_2 = np.array(l_img < self.threshold)
            binary_img_array = np.logical_and(binary_img_array_1, binary_img_array_2) * 255
        elif self.name == "GNB":  # Gaussian Naive Bayes
            marked_thumbnail = np.array(wsi_thumb_img)
            gnb_model = self.get_gnb_model()
            cal = gnb_model.predict_proba(marked_thumbnail.reshape(-1, 3))
            cal = cal.reshape(marked_thumbnail.shape[0], marked_thumbnail.shape[1], 2)
            binary_img_array = cal[:, :, 1] > self.threshold
        elif self.name == "SVM":  # Support vector machine
            marked_thumbnail = np.array(wsi_thumb_img)
            svm_model = self.get_gnb_model()
            cal = svm_model.predict_proba(marked_thumbnail.reshape(-1, 3))
            cal = cal.reshape(marked_thumbnail.shape[0], marked_thumbnail.shape[1], 2)
            binary_img_array = cal[:, :, 1] > self.threshold
        else:
            raise Exception("Undefined model")
        # plt.imshow(binary_img_array)
        # plt.show()
        if open_operation:
            binary_img_array = ndimage.binary_opening(binary_img_array, structure=np.ones((5, 5))).astype(
                binary_img_array.dtype)  # open operation
        return binary_img_array


# example
if __name__ == '__main__':
    print("see auto_wsi_matcher.py for examples")


