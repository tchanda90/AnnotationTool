import sys
import os
import pandas as pd
import torch
from torchvision import transforms
from skimage import io
from models import ArtfifactDetectorSingle
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QFileDialog
from PyQt5.QtGui import QPixmap


class Setup(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = ''
        self.annotations_file = ''
        uic.loadUi('setup.ui', self)

        self.button_start.clicked.connect(self.start_annotation)

        self.button_img_path.clicked.connect(self.select_folder)
        self.button_ann_file.clicked.connect(self.select_file)

    def select_folder(self):
        self.lineEdit_img_path.setText(QFileDialog.getExistingDirectory(self, 'Select Folder'))

    def select_file(self):
        self.lineEdit_ann_file.setText(QFileDialog.getOpenFileName(self, 'Select File')[0])

    def start_annotation(self):

        if self.lineEdit_img_path.text() == "":
            return

        self.image_folder = self.lineEdit_img_path.text()

        self.annotations_file = self.lineEdit_ann_file.text()

        annotator = Annotator(self.image_folder, self.annotations_file)
        widget.addWidget(annotator)
        widget.setCurrentIndex(widget.currentIndex()+1)


class Annotator(QWidget):
    def __init__(self, image_folder, annotations_file):
        super().__init__()

        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.list_of_images = []
        self.num_images = 0
        self.current_img_index = 0

        self.annotations = {}

        self.models = {'ruler': ArtfifactDetectorSingle(), 'border': ArtfifactDetectorSingle(),
                       'stain': ArtfifactDetectorSingle()}
        self.models['ruler'].load_state_dict(torch.load(os.path.join('models', 'ruler_detector.pt'), map_location="cpu"))
        self.models['border'].load_state_dict(torch.load(os.path.join('models', 'border_detector.pt'), map_location="cpu"))
        self.models['stain'].load_state_dict(torch.load(os.path.join('models', 'stain_detector.pt'), map_location="cpu"))

        self.models['ruler'].eval()
        self.models['border'].eval()
        self.models['stain'].eval()

        uic.loadUi('annotate.ui', self)

        self.button_next.clicked.connect(self.show_next_image)
        self.button_prev.clicked.connect(self.show_prev_image)
        self.button_save.clicked.connect(self.save_annotations)

        self.init()

    def init(self):

        files = sorted(os.listdir(self.image_folder))
        for file in files:
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                self.list_of_images.append(file)

        # If an annotations file already exists, read the annotations from it and remove those images
        #  from the current list of images. Then populate the current annotations dict with those annotations.
        if os.path.isfile(self.annotations_file):
            annotations_df = pd.read_csv(self.annotations_file, na_filter=False).set_index('image')
            annotated_images = annotations_df.index

            for img in annotated_images:
                self.list_of_images.remove(img)

            self.annotations = annotations_df.to_dict(orient='index')
        else:
            self.annotations_file = os.path.join(self.image_folder, "annotations.csv")
            df = self.create_empty_df()
            df.to_csv(self.annotations_file, index=False)

        self.list_of_images = sorted(self.list_of_images)
        self.num_images = len(self.list_of_images)

        self.show_image()

    def create_empty_df(self):
        return pd.DataFrame({'ruler': [], 'border': [], 'stain': [],
                             'subtle_ruler': [], 'subtle_border': [], 'subtle_stain': [], 'comments': []})

    def image_loader(self, transform, img_path):
        image = io.imread(img_path)
        image = transform(image)
        image = image.detach().unsqueeze(0)
        return image

    def predict_image(self, img_path):

        channels, height, width = (3, 300, 300)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((width, height)),
                                        transforms.ToTensor()])

        image = self.image_loader(transform, img_path)

        predictions = {'ruler': self.models['ruler'].forward(image).detach().round().numpy()[0][0],
                       'border': self.models['border'].forward(image).detach().round().numpy()[0][0],
                       'stain': self.models['stain'].forward(image).detach().round().numpy()[0][0]}
        return predictions

    def set_predictions(self, predictions):
        if predictions['ruler'] == 1:
            self.checkBox_class_ruler.setChecked(True)
        if predictions['border'] == 1:
            self.checkBox_class_border.setChecked(True)
        if predictions['stain'] == 1:
            self.checkBox_class_stain.setChecked(True)

    def show_image(self):

        img_path = os.path.join(self.image_folder, self.list_of_images[self.current_img_index])

        predictions = self.predict_image(img_path)
        self.set_predictions(predictions)

        self.label_img_name.setText(self.list_of_images[self.current_img_index])

        pixmap = QPixmap(img_path)
        self.label_image.setPixmap(pixmap)

        widget.update()

    def show_next_image(self):

        ruler = 0
        border = 0
        stain = 0
        s_ruler = 0
        s_border = 0
        s_stain = 0

        comments = self.lineEdit_comments.text()
        if self.checkBox_class_ruler.isChecked():
            ruler = 1
        if self.checkBox_class_border.isChecked():
            border = 1
        if self.checkBox_class_stain.isChecked():
            stain = 1
        if self.checkBox_class_subtle_ruler.isChecked():
            s_ruler = 1
        if self.checkBox_class_subtle_border.isChecked():
            s_border = 1
        if self.checkBox_class_subtle_stain.isChecked():
            s_stain = 1

        img = self.list_of_images[self.current_img_index]
        self.annotations[img] = {'ruler': ruler, 'border': border, 'stain': stain, 'subtle_ruler': s_ruler,
                                 'subtle_border': s_border, 'subtle_stain': s_stain, 'comments': comments}

        # If end of images reached. Display a message and exit this function.
        if self.current_img_index + 1 == self.num_images:
            self.label_img_save.setText('No more images left. Click save progress and close the app.')
            return

        self.current_img_index += 1

        img = self.list_of_images[self.current_img_index]
        self.set_checkboxes(img)

        self.label_img_save.clear()

        self.show_image()

    def show_prev_image(self):

        # If clicked previous button on first image, display a message and exit this function
        if self.current_img_index == 0:
            self.label_img_save.setText('No previous images.')
            return

        self.current_img_index -= 1

        img = self.list_of_images[self.current_img_index]

        self.set_checkboxes(img)

        self.label_img_save.clear()

        self.show_image()

    def save_annotations(self):
        annotations_df = pd.DataFrame(self.annotations).T.reset_index().rename({'index': 'image'}, axis=1)
        annotations_df.to_csv(self.annotations_file, index=False)

        self.label_img_save.setText('Annotations Saved at ' + self.annotations_file)

    def set_checkboxes(self, img):

        if img not in self.annotations:
            self.checkBox_class_ruler.setChecked(False)
            self.checkBox_class_border.setChecked(False)
            self.checkBox_class_stain.setChecked(False)
            self.checkBox_class_subtle_ruler.setChecked(False)
            self.checkBox_class_subtle_border.setChecked(False)
            self.checkBox_class_subtle_stain.setChecked(False)
            self.lineEdit_comments.clear()
            return

        if self.annotations[img]['ruler']:
            self.checkBox_class_ruler.setChecked(True)
        else:
            self.checkBox_class_ruler.setChecked(False)
        if self.annotations[img]['border']:
            self.checkBox_class_border.setChecked(True)
        else:
            self.checkBox_class_border.setChecked(False)
        if self.annotations[img]['stain']:
            self.checkBox_class_stain.setChecked(True)
        else:
            self.checkBox_class_stain.setChecked(False)

        if self.annotations[img]['subtle_ruler']:
            self.checkBox_class_subtle_ruler.setChecked(True)
        else:
            self.checkBox_class_subtle_ruler.setChecked(False)
        if self.annotations[img]['subtle_border']:
            self.checkBox_class_subtle_border.setChecked(True)
        else:
            self.checkBox_class_subtle_border.setChecked(False)
        if self.annotations[img]['subtle_stain']:
            self.checkBox_class_subtle_stain.setChecked(True)
        else:
            self.checkBox_class_subtle_stain.setChecked(False)

        if self.annotations[img]['comments'] != '':
            self.lineEdit_comments.setText(self.annotations[img]['comments'])
        else:
            self.lineEdit_comments.clear()


if __name__ == "__main__":

    w = 1600
    h = 900

    app = QApplication(sys.argv)

    widget = QStackedWidget()

    setup = Setup()
    widget.addWidget(setup)
    widget.resize(w, h)
    widget.show()
    sys.exit(app.exec())
