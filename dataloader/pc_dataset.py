import os
import yaml
import numpy as np

from PIL import Image
from torch.utils import data
from pathlib import Path
from nuscenes.utils import splits

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


@register_dataset
class SemanticKITTI(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)
            calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-22:-20])]

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]


@register_dataset
class MCDNTU(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder), 'inL_bin']), num_vote)
            calib_path = "./dataloader/calib_mcdntu.txt"
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["Ps"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        if seg_num_per_class is not None:
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['Ps'] = calib_all['Ps'].reshape(3, 4)  # 3x4 projection matrix for left camera
        
        calib_out['Tr_c2w'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr_c2w'][:3, :4] = calib_all['T_CM'].reshape(3, 4)
        calib_out['Tr_w2c'] = np.linalg.inv(calib_out['Tr_c2w'])
        
        calib_out['Tr_l2w'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr_l2w'][:3, :4] = calib_all['T_LV'].reshape(3, 4)
        
        calib_out['Tr'] = calib_out['Tr_w2c'] @ calib_out['Tr_l2w']

        return calib_out

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('inL_bin', 'label')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            # annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            # annotated_data[annotated_data == 21] = 255

        frame_id = self.im_idx[index][-8:-4]
        image_file = self.im_idx[index].replace('inL_bin', 'image').replace('.bin', '.png').replace(frame_id, "00" + frame_id)
        try:
            image = Image.open(image_file)
        except:
            image = None
        seq = self.im_idx[index].split('/')[-3]
        proj_matrix = self.proj_matrix[seq]

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]


@register_dataset
class nuScenes(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                if imageset == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']

        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])
                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'cam_token': cam_token}
                    )

    def __getitem__(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor',
                                        pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor',
                                      cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img'] = image
        data_dict['calib_infos'] = calib_infos
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)

        return data_dict, lidar_sample_token


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    config = {
        "dataset_params":{
            "label_mapping": "/data1/haozhi/code/2DPASS/config/label_mapping/mcd-ntu.yaml",
            'seg_labelweights': None
        },
        'train_params': {}
    }
    dataset = MCDNTU(
        config, 
        '/data1/haozhi/code/2DPASS/dataset/MCD_NTU',
        'train')
    
    # Count class-wise point number
    # points_per_class = np.zeros(24, int)
    # for sample in tqdm(dataset):
    #     labels = sample[0]['labels']
    #     points_per_class += np.bincount(labels[labels != 255], minlength=24)
    
    # print("Point num per class: ")
    # for i in range(points_per_class.shape[0]):
    #     print("- {}".format(points_per_class[i]))
    
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind
    
    def depth_color(val, min_d=0, max_d=120):
        """ 
        print Color(HSV's H value) corresponding to distance(m) 
        close distance = red , far distance = blue
        """
        np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 
    
    def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5, save=None):
        # depth = normalize_depth(depth, d_min=3., d_max=50.)
        # depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
        # colors = []
        # for depth_val in depth.tolist():
        #     colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
        colors = depth_color(depth).tolist()
        fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
        # ax5.imshow(np.full_like(img, 255))
        ax.imshow(img)
        ax.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.7, s=point_size)

        ax.axis('off')

        if show:
            plt.show()
            
        if save is not None:
            plt.savefig(save)
            plt.close()
    
    for sample, _ in dataset:
        img = sample['img']
        xyz = sample['xyz']
        proj_matrix = sample['proj_matrix']
        labels = sample['labels']
        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = select_points_in_frustum(img_points, 0, 0, *img.size)
        keep_idx[keep_idx] = keep_idx_img_pts
        
        depth = np.linalg.norm(xyz, axis=1)
        img = np.array(img)
        draw_points_image_depth(img, img_points, depth, save="./mcdntu_point2img.png")
        
        input("Press Enter to continue...")
        