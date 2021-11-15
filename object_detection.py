
!pip install icevision[all]

!pip install torchtext==0.9.1
!pip install folium==0.2.1
!pip install torchvision==0.9.1

from icevision.all import *
import pandas as pd
from sklearn.model_selection import train_test_split

dir = "/711Ass3/"

data = pd.read_csv( dir + 'train_set.csv')
data, valid = train_test_split(data, test_size=0.125, stratify = data['label'])
test = pd.read_csv( dir + 'test_set.csv')

template_record = ObjectDetectionRecord()
Parser.generate_template(template_record)

class DataParser(Parser):
    def __init__(self, template_record, data_dir, df_data):
        super().__init__(template_record=template_record)

        self.data_dir = data_dir
        self.df = df_data
        self.class_map = ClassMap(list(self.df['label'].unique()))

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o

    def __len__(self) -> int:
        return len(self.df)

    def record_id(self, o) -> Hashable:
        return o.file_name

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.data_dir + o.file_name)
            record.set_img_size(ImgSize(width=o.width, height=o.height))
            record.detection.set_class_map(self.class_map)

        record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
        record.detection.add_labels([o.label])

dataSplitter = RandomSplitter([1.0,0.0])
data_dir =  dir + "Training/"
parser = DataParser(template_record, data_dir, valid)
valid_records, nothing = parser.parse(data_splitter = dataSplitter)
parser = DataParser(template_record, data_dir, data)
train_records, nothing = parser.parse(data_splitter = dataSplitter)
parser = DataParser(template_record, dir + "Testing/", test)
test_records, nothing = parser.parse(data_splitter = dataSplitter)

image_size = 384

train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize((0.4717, 0.5291, 0.3492), (0.2425, 0.2290, 0.2500))])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size),tfms.A.Normalize((0.4717, 0.5291, 0.3492), (0.2425, 0.2290, 0.2500))])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)
test_ds = Dataset(test_records, valid_tfms)

model_type = models.ross.efficientdet
extra_args = {}
backbone = model_type.backbones.d0
extra_args['img_size'] = image_size
model = model_type.model(backbone=backbone(pretrained=False), num_classes=len(parser.class_map), **extra_args)

#model_type = models.torchvision.faster_rcnn
#backbone = model_type.backbones.resnet18_fpn
#model = model_type.model(backbone=backbone(pretrained=False), num_classes=len(parser.class_map))

train_dl = model_type.train_dl(train_ds, batch_size=32, num_workers=0, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=32,num_workers=0, shuffle=False)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox, iou_thresholds = [0.5])]

learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

learn.fit(20)

infer_dl = model_type.infer_dl(test_ds, batch_size=32)
preds = model_type.predict_from_dl(model=model, infer_dl=infer_dl, keep_images=True)

metric = COCOMetric(metric_type=COCOMetricType.bbox, iou_thresholds = [0.5], print_summary = True)
metric.accumulate(preds)
fin = metric.finalize()
