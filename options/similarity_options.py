from .base_options import BaseOptions


class SimilarityOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # Cross similarity
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--pred_T', action='store_true', help='if true, save Teacher model outputs and index for train datasets')
        self.parser.add_argument('--feature_T', action='store_true', help='if ture, save Teacher model feature map outputs and index for train datasets')
        self.parser.add_argument('--T_path', type=str, help='Path of teacher network')

        self.isTrain = False
        self.isKD = False
