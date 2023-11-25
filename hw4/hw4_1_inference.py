import os, sys, random, json
from tqdm import tqdm, trange

import imageio
import numpy as np

import torch


from hw4_1_dvgo_lib import utils, dvgo

# 
# json_dir = '/home/kszuyen/DLCV/hw4-kszuyen/hw4_data/hotdog/transforms_val.json'
# output_dir = '/home/kszuyen/DLCV/hw4-kszuyen/hw4_1_output'

json_dir = sys.argv[1]
output_dir = sys.argv[2]
#

# model_dir = "/home/kszuyen/DLCV/hw4-kszuyen/DirectVoxGO/.logs/settings_2/dvgo_hotdog/fine_last.tar"
model_dir = ".logs/settings_1/dvgo_hotdog/fine_last.tar"

def seed_everything(seed=777):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(json_dir):

    with open(json_dir, 'r') as fp:
        metas = json.load(fp)

    fnames = []
    poses = []
    
    for frame in metas['frames']:
        fnames.append(frame['file_path'].split('/')[-1])
        poses.append(np.array(frame['transform_matrix']).astype(np.float32))

    H, W = 800, 800
    camera_angle_x = float(metas['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    return poses, [H, W, focal], fnames
def load_data(json_dir):
    poses, hwf, fnames = load_blender_data(json_dir)

    near, far = 2., 6.
    H, W, focal = hwf

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    # render_poses = render_poses[...,:4]

    data_dict = dict(
        # hwf=hwf, HW=HW, Ks=Ks,
        Ks=Ks,
        near=near, far=far,
        poses=poses, fnames=fnames
    )
    data_dict['poses'] = torch.Tensor(data_dict['poses'])

    return data_dict

@torch.no_grad()
def render_viewpoints(model, render_poses, Ks, ndc, 
                        render_kwargs, fnames, output_dir):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''

    rgbs = []
    depths = []
    bgmaps = []

    for i, c2w in enumerate(tqdm(render_poses)):

        # H, W = HW[i]
        H, W = 800, 800
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=False, flip_y=False)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

    for i in trange(len(rgbs)):
        rgb8 = utils.to8b(rgbs[i])
        filename = os.path.join(output_dir, f'{fnames[i]}.png')
        imageio.imwrite(filename, rgb8)

# init environment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
seed_everything()

# load poses / camera settings
data_dict = load_data(json_dir=json_dir)

# load model for rendering
model_class = dvgo.DirectVoxGO

model = utils.load_model(model_class, model_dir).to(device)
render_viewpoints_kwargs = {
    'model': model,
    'ndc': False,
    'render_kwargs': {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1,
        'stepsize': 0.5,
        'inverse_y': False,
        'flip_x': False,
        'flip_y': False,
        'render_depth': True,
    },
}
render_viewpoints(
    render_poses=data_dict['poses'],
    Ks=data_dict['Ks'],
    fnames=data_dict['fnames'],
    output_dir=output_dir,
    **render_viewpoints_kwargs
)

