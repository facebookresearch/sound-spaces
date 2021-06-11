from collections import defaultdict
import attr

from scipy.spatial import cKDTree
import numpy as np
from numpy.linalg import norm


SCENE_SPLITS = {
    'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
              'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
              'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
              'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
              'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
              'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
              '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
              'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
              'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
              '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
    'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
            'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
    'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
             'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
             'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
}
SCENE_SPLITS['train_distractor'] = SCENE_SPLITS['train']
SCENE_SPLITS['val_distractor'] = SCENE_SPLITS['val']
SCENE_SPLITS['test_distractor'] = SCENE_SPLITS['test']

MPCAT40_CATEGORY_INDICES = [3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 18, 19, 20, 22, 23, 25, 26, 27, 33, 34, 38]


CATEGORY_INDEX_MAPPING = {
            'chair': 0,
            'table': 1,
            'picture': 2,
            'cabinet': 3,
            'cushion': 4,
            'sofa': 5,
            'bed': 6,
            'chest_of_drawers': 7,
            'plant': 8,
            'sink': 9,
            'toilet': 10,
            'stool': 11,
            'towel': 12,
            'tv_monitor': 13,
            'shower': 14,
            'bathtub': 15,
            'counter': 16,
            'fireplace': 17,
            'gym_equipment': 18,
            'seating': 19,
            'clothes': 20
        }


@attr.s
class Object:
    object_index = attr.ib(converter=int)
    region_index = attr.ib(converter=int)
    category_index = attr.ib(converter=int)
    px = attr.ib(converter=float)
    py = attr.ib(converter=float)
    pz = attr.ib(converter=float)
    a0x = attr.ib(converter=float)
    a0y = attr.ib(converter=float)
    a0z = attr.ib(converter=float)
    a1x = attr.ib(converter=float)
    a1y = attr.ib(converter=float)
    a1z = attr.ib(converter=float)
    r0 = attr.ib(converter=float)
    r1 = attr.ib(converter=float)
    r2 = attr.ib(converter=float)


class HouseReader:
    """
        The .house file has a sequence of ascii lines with fields separated by spaces in the following format:

        H name label #images #panoramas #vertices #surfaces #segments #objects #categories #regions #portals #levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        L level_index #regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
        P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
        S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
        P name  panorama_index region_index 0  px py pz  0 0 0 0 0
        I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
        C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
        O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0
        E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0

        where xxx_index indicates the index of the xxx in the house file (starting at 0),
        #xxxs indicates how many xxxs will appear later in the file that back reference (associate) to this entry,
        (px,py,pz) is a representative position, (nx,ny,nz) is a normal direction,
        (xlo, ylo, zlo, xhi, yhi, zhi) is an axis-aligned bounding box,
        camera_index is in [0-5], yaw_index is in [0-2],a
        (e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33) are the extrinsic matrix of a camera,
        (i00 i01 i02  i10 i11 i12 i20 i21 i22) are the intrinsic matrix for a camera,
        (px, py, pz, a0x, a0y, a0z, a1x, a1y, a1z, r0, r1, r2) define the center, axis directions, and radii of an oriented bounding box,
        height is the distance from the floor, and
        0 is a value that can be ignored.

        The extent of each region is defined by a prism with its vertical extent dictated by its height and
        its horizontal cross-section dictated by the counter-clockwise set of polygon vertices associated
        with each surface associated with the region.

        The extent of each object is defined by the oriented bounding box of the 'O' command.
        The set of faces associated with each segment are ones whose 'face_material' field
        in the xxx.ply file (described next) matches the segment 'id' in the 'S' command.
    """
    def __init__(self, house_file):
        self.data = defaultdict(list)
        self.category_index2mpcat40_index = dict()
        self.category_index2mpcat40_name = dict()

        with open(house_file, 'r') as fo:
            annotations = fo.readlines()
        for line in annotations[1:]:
            tokens = line.split()
            if tokens[0] == 'C':
                category_index = int(tokens[1])
                mpcat40_index = int(tokens[4])
                mpcat40_name = tokens[5]
                self.category_index2mpcat40_index[category_index] = mpcat40_index
                self.category_index2mpcat40_name[category_index] = mpcat40_name
            elif tokens[0] == 'O':
                obj = Object(*tokens[1:16])
                self.data[tokens[0]].append(obj)
            else:
                self.data[tokens[0]].append(tokens[1:])

    def find_objects_with_mpcat40_index(self, mpcat40_index):
        found_objects = list()
        for obj in self.data['O']:
            if obj.category_index == -1:
                #                 logging.warning('Category index: {}'.format(obj.category_index))
                continue
            elif self.category_index2mpcat40_index[obj.category_index] == mpcat40_index:
                found_objects.append(obj)
        return found_objects

    def find_objects_with_mpcat40_indices(self):
        objects = []
        for index in MPCAT40_CATEGORY_INDICES:
            objects += self.find_objects_with_mpcat40_index(index)
        return objects

    def find_objects_close_to(self, objects, points, threshold=1):
        points = np.array(points)
        kd_tree = cKDTree(points[:, [0, 2]])

        num_object = 0
        if len(objects) > 0:
            obj_pos = np.array([(obj.px, -obj.py) for obj in objects])
            d, _ = kd_tree.query(obj_pos)
            num_object = sum(d < threshold)

        return num_object

    def find_points_in_bbx(self, points, objects, tol=1):
        points = np.array(points)
        points = np.stack([points[:, 0], -points[:, 2], points[:, 1] + 1.5], axis=-1)

        num_object = 0
        if len(objects) > 0:
            for obj in objects:
                v = points - np.array([obj.px, obj.py, obj.pz])
                a0 = np.array([obj.a0x, obj.a0y, obj.a0z])
                a1 = np.array([obj.a1x, obj.a1y, obj.a1z])
                a2 = np.cross(a0, a1) / np.linalg.norm(np.cross(a0, a1))
                d0 = np.inner(v, a0)
                d1 = np.inner(v, a1)
                d2 = np.inner(v, a2)
                inside_bbx = (abs(d0) < obj.r0 + tol) & (abs(d1) < obj.r1 + tol) & (abs(d2) < obj.r2 + tol)
                #    print('{} points are near object, distances: {}/{}/{}'.format(sum(inside_bbx), d0, d1, d2))
                num_object += any(inside_bbx)

        return num_object

    def compute_object_to_category_index_mapping(self):
        objects = self.data['O']
        mapping = dict()
        for obj in objects:
            if obj.category_index == -1:
                mpcat40_index = -1
            else:
                mpcat40_index = self.category_index2mpcat40_index[obj.category_index]
            mapping[obj.object_index] = mpcat40_index

        return mapping

