from types import SimpleNamespace

class Config:
    def __init__(self):
        self.TRAINER = SimpleNamespace()
        self.TRAINER.MAPLE = SimpleNamespace()
        self.TRAINER.MAPLE.N_CTX = 16
        self.TRAINER.MAPLE.CSC = False
        self.TRAINER.MAPLE.CLASS_TOKEN_POSITION = "end"
        self.TRAINER.MAPLE.PREC = "fp16"

        self.MODEL = SimpleNamespace()
        self.MODEL.BACKBONE = SimpleNamespace()
        self.MODEL.BACKBONE.NAME = "ViT-B/32"

        self.DATASET = SimpleNamespace()
        self.DATASET.NAME = "twitter2015"

        self.MODEL.INIT_WEIGHTS = None

        self.OPTIM = SimpleNamespace()
        self.OPTIM.NAME = "sgd"
        self.OPTIM.LR = 0.002
        self.OPTIM.WEIGHT_DECAY = 0.0005
        self.OPTIM.MOMENTUM = 0.9

        self.TRAIN = SimpleNamespace()
        self.TRAIN.BATCH_SIZE = 32

        self.TEST = SimpleNamespace()
        self.TEST.BATCH_SIZE = 100

cfg = Config()