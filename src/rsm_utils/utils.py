import os
import glob
import tifffile
import numpy as np
from tqdm import trange
import xrayutilities as xu

KAPPA_GEOMETRY = {
    'tth': '+z',
    'th': '+z',
    'chi': '-x',
    'phi': '+z',
    'detx': '+y', # direct beam pos
    'dety': '+z',
    'kin': '+x',
    'pixelsize' : 11e-6,
    'D' : 0.2193, # detector distance
}

def map_hkl(tiff_fname, geometry=KAPPA_GEOMETRY):

    def rot_mat(axis, deg):
        '''
        Returns matrix of rotation about x-, y-, or z-axis
        '''
        axis = axis.strip().lower()
        rad = np.deg2rad(deg)
        if axis.lower()=='x' or axis.lower()=='+x':
            r = np.array([[1, 0, 0],[0, np.cos(rad), -np.sin(rad)],[0, np.sin(rad), np.cos(rad)]]) 
        elif axis.lower()=='y' or axis.lower()=='+y':
            r = np.array([[np.cos(rad), 0, np.sin(rad)],[0, 1, 0],[-np.sin(rad), 0, np.cos(rad)]]) 
        elif axis.lower()=='z' or axis.lower()=='+z':
            r = np.array([[np.cos(rad), -np.sin(rad), 0],[np.sin(rad), np.cos(rad), 0],[0, 0, 1]])
        elif axis.lower()=='-x':
            r = np.array([[1, 0, 0],[0, np.cos(rad), np.sin(rad)],[0, -np.sin(rad), np.cos(rad)]]) 
        elif axis.lower()=='-y':
            r = np.array([[np.cos(rad), 0, -np.sin(rad)],[0, 1, 0],[np.sin(rad), 0, np.cos(rad)]]) 
        elif axis.lower()=='-z':
            r = np.array([[np.cos(rad), np.sin(rad), 0],[-np.sin(rad), np.cos(rad), 0],[0, 0, 1]])
        else:
            raise ValueError(f'axis must be x, y, z, +x, +y, +z, -x, -y, or -z; got {axis!r}')
        return r
        
    img = tifffile.imread(tiff_fname)
    with tifffile.TiffFile(tiff_fname) as tif:
        metadata_dict = {}
        for tag in tif.pages[0].tags:
            try:
                k, v = tag.value.split(':')
                k = k.strip()
                v = v.strip()
            except Exception:
                k, v = tag.name, tag.value
            metadata_dict[k] = v
    
    lattice = np.array([float(metadata_dict[f'Sample_lattice_{i}']) for i in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
    UB = np.array([float(metadata_dict[f'UB{i+1}']) for i in range(9)]).reshape(3, 3)
    tth = float(metadata_dict['Kappa_tth'])
    th = float(metadata_dict['Kappa_th'])
    chi = float(metadata_dict['Kappa_chi'])
    phi = float(metadata_dict['Kappa_phi'])
    hv = float(metadata_dict['mon_rbv'])
    beam_cenX = int(metadata_dict['Direct_beam_pixelX'])
    beam_cenY = int(metadata_dict['Direct_beam_pixelY'])
    
    phi_rot = rot_mat(geometry['phi'], phi)
    chi_rot = rot_mat(geometry['chi'], chi)
    th_rot = rot_mat(geometry['th'], th)
    tth_rot = rot_mat(geometry['tth'], tth)
    abc_in_lab = th_rot @ chi_rot @ phi_rot @ UB 

    k_wl = 2*np.pi*hv/12398 # Angstrom^-1

    pix_x, pix_y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    d_pix_x = (pix_x - beam_cenX)*geometry['pixelsize']
    d_pix_y = (pix_y - beam_cenY)*geometry['pixelsize']
    d_pix_z = np.ones_like(d_pix_x)*geometry['D']
    d_pix_lst = [d_pix_x, d_pix_y, d_pix_z]

    def gen_xyz_order(detx, dety, kin, d_pix_lst):
        (sign0, axis0) = (int(detx[0]+'1'), detx[1]) if len(detx) == 2 else (1, detx[0])
        (sign1, axis1) = (int(dety[0]+'1'), dety[1]) if len(dety) == 2 else (1, dety[0])
        (sign2, axis2) = (int(kin[0]+'1'), kin[1]) if len(kin) == 2 else (1, kin[0])
        if [axis0, axis1, axis2] == ['x', 'y', 'z']:
            return [sign0*d_pix_lst[0], sign1*d_pix_lst[1], sign2*d_pix_lst[2]]
        elif [axis0, axis1, axis2] == ['x', 'z', 'y']:
            return [sign0*d_pix_lst[0], sign2*d_pix_lst[2], sign1*d_pix_lst[1]]
        elif [axis0, axis1, axis2] == ['y', 'x', 'z']:
            return [sign1*d_pix_lst[1], sign0*d_pix_lst[0], sign2*d_pix_lst[2]]
        elif [axis0, axis1, axis2] == ['y', 'z', 'x']:
            return [sign2*d_pix_lst[2], sign0*d_pix_lst[0], sign1*d_pix_lst[1]]
        elif [axis0, axis1, axis2] == ['z', 'x', 'y']:
            return [sign1*d_pix_lst[1], sign2*d_pix_lst[2], sign0*d_pix_lst[0]]
        elif [axis0, axis1, axis2] == ['z', 'y', 'x']:
            return [sign2*d_pix_lst[2], sign1*d_pix_lst[1], sign0*d_pix_lst[0]]
        else:
            raise Exception('detx, dety, kin should be along different directions.')

    dx, dy, dz = gen_xyz_order(geometry['detx'], geometry['dety'], geometry['kin'], d_pix_lst)

    pix_labpos = np.stack([dx, dy, dz], axis=2)
    pix_labpos_at_tth = (tth_rot[None, None, :, :] @ pix_labpos[:, :, :, None]).squeeze(-1)
    q_map = k_wl*(pix_labpos_at_tth/np.linalg.norm(pix_labpos_at_tth, axis=-1, keepdims=True) - np.array([1, 0, 0]))
    hkl_map = (np.linalg.inv(abc_in_lab)[None, None, :, :] @ q_map[:, :, :, None]).squeeze(-1)
    return img, hkl_map

def build_rsm(path, scan_lst, nx=200, ny=200, nz=200):
    hkl_min_lst = []
    hkl_max_lst = []
    for scan_num in scan_lst:
        scan_folder = os.path.join(path, f'{scan_num:04d}')
        h_min, h_max, k_min, k_max, l_min, l_max = find_hkl_range(scan_folder)
        hkl_min_lst.append([h_min, k_min, l_min])
        hkl_max_lst.append([h_max, k_max, l_max])
    h_min_all, k_min_all, l_min_all = np.min(hkl_min_lst, axis=0)
    h_max_all, k_max_all, l_max_all = np.max(hkl_max_lst, axis=0)

    gridder3d = xu.Gridder3D(nx, ny, nz)

    gridder3d.xmin = h_min_all
    gridder3d.xmax = h_max_all
    gridder3d.ymin = k_min_all
    gridder3d.ymax = k_max_all
    gridder3d.zmin = l_min_all
    gridder3d.zmax = l_max_all

    gridder3d.keep_data = True

    gridder3d(
        np.array([h_min_all, h_max_all]),
        np.array([k_min_all, k_max_all]),
        np.array([l_min_all, l_max_all]),
        np.zeros(2)
    )

    try:
        for scan_num in scan_lst:
            print(f'Loading images in scan {scan_num}')
            scan_folder = os.path.join(path, f'{scan_num:04d}')
            fpath_lst = sorted(glob.glob(os.path.join(scan_folder,'*.TIFF')))
            for i in trange(len(fpath_lst)):
                fpath = fpath_lst[i]
                img, hkl_map = map_hkl(fpath)
                gridder3d(
                    hkl_map[:, :, 0].flatten(),
                    hkl_map[:, :, 1].flatten(),
                    hkl_map[:, :, 2].flatten(),
                    img.flatten()
                )
        H_grid = gridder3d.xmatrix.copy()
        K_grid = gridder3d.ymatrix.copy()
        L_grid = gridder3d.zmatrix.copy()
        I_grid = gridder3d.data.copy()
        return H_grid, K_grid, L_grid, I_grid
    except Exception as e:
        print(e)
        raise



def find_hkl_range(scan_folder, img_nums=[0, -1]):
    try:
        fpath_lst = sorted(glob.glob(os.path.join(scan_folder,'*.TIFF')))
        hkl_min_lst = []
        hkl_max_lst = []
        print(f'Checking HKL ranges in {scan_folder}...')
        for idx in img_nums:
            img, hkl_map = map_hkl(fpath_lst[idx])
            hkl_min_lst.append(np.min(hkl_map, axis=(0, 1)))
            hkl_max_lst.append(np.max(hkl_map, axis=(0, 1)))
        h_min, k_min, l_min = np.min(hkl_min_lst, axis=0)
        h_max, k_max, l_max = np.max(hkl_max_lst, axis=0)
        return h_min, h_max, k_min, k_max, l_min, l_max
    except Exception as e:
        print(f'ERROR! find_xyz_range: {e}')
        raise