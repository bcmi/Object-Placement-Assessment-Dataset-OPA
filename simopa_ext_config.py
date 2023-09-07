import os

class Config(object):
    class_num = 2
    refer_class_num = 1601
    img_size = 256
    batch_size  = 128
    num_workers = 8
    binary_mask_size = 64
    geometric_feature_dim = 256
    roi_align_size = 3
    global_feature_size = 8
    attention_dim_head = 64
    
    backbone = 'resnet18'
    gpu_id = 1
    pretrained_model_path = './pretrained_models'

    # train
    base_lr = 1e-4
    lr_milestones = [10, 16]
    lr_gamma = 0.1
    epochs = 20
    eval_freq = 1
    save_freq = 1
    display_freq = 10

    # config for baseline model
    refer_num = 5
    attention_head = 16
    without_mask = False
    relation_method_str = ['roi_align', 'only_target_box', 'average_all_boxes',
                           'without_geometry', 'simple_geometry', 'proposed_relation']
    relation_method = None # one of [None,0,1,2,3,4,5]
    attention_method_str = ['only_attention_score', 'without_attention_score', 'proposed_attention']
    attention_method = None # one of [None,0,1,2]
    without_global_feature = False
    
    # Save path 
    prefix = backbone
    if without_mask:
        prefix += '+without_mask'

    exp_root = os.path.join(os.getcwd(), './experiments/simopa_ext/')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)

    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiments directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

opt = Config()
