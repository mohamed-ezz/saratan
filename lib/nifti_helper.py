__author__ = 'mbickel'

import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

import os
import sys
import random

import imp
f3 = imp.load_source('f3', os.path.normpath('../lib/fire3db.py'))

##
#
# Methods requiring initialization of database.
#
##

class NiftiDBHelper:
    """Helper class to get niftis or numpy tensors registered in fire3 SQLite database."""

    def __init__(self, path, type='train'):
        self.set_db(path)

        self.__table = None

        if type == 'train':
            self.__table = f3.Segmented
        elif type == 'test':
            #self.__table = f3.TestSet
            pass
        else:
            raise ValueError('Unknown type (' + str(type) + ').')

        self.count = self.q_total()
        self.row = None

        self.shuffled_uids = self.get_shuffled_uids()

    def __del__(self):
        f3.close_db()

    def set_db(self, path):
        """
        Set a sqlite db for this package.
        :param path: Path to sqlite db with segmented table (see fire3db package)
        """
        f3.init_db(os.path.normpath(path))

    def get_shuffled_uids(self):
        suids = []

        for uid in f3.Segmented.select(f3.Segmented.uid):
            suids.append(uid.uid)

        random.shuffle(suids)

        return suids

    def get_row(self):
        if self.row != None:
            return self.row

    def q_first_row(self):
        self.row = f3.Segmented.select().first()
        return self.row

    def q_x_row(self, nb):
        """Query Xth row in Segmented table. Start with 0"""
        if nb > (self.q_total() - 1):
            raise IndexError

        self.row = f3.Segmented.select().offset(nb).limit(1).first()
        return self.row

    def q_shuffled_row(self):
        self.row = f3.Segmented.select().where(f3.Segmented.uid == self.shuffled_uids.pop()).limit(1).first()
        return self.row

    def q_total(self):
        """Get the count of rows in the Segmented table."""
        return f3.Segmented.select().count()

    def q_random(self):
        """Query a random row"""
        self.row = self.q_x_row(random.randint(0, (self.q_total() - 1)))
        return self.row

    def q_all_rows(self):
        """Lazy iterator over all rows in Segmented table.
        :return:Lazy segmented iterator.
        """
        return f3.Segmented.select().iterator()

    def get_current_row(self):
        """Get the currently cached row or None if no query was done yet."""
        return self.row

    def get_ct_img(self):
        """Get currently cached CT-Image"""
        return load_nifti(self.row.ctNiftiFilename)

    def get_ct_img_data(self):
        """Get image data of currently cached CT-Image as numpy tensor"""
        return get_img_data(self.get_ct_img())

    def get_ct_img_dim(self):
        """
        Get dimension of currently cached CT-Image
        :return: x range, y range, z range (3 return vals)
        """
        return get_dim(self.get_ct_img())

    def get_seg_img(self):
        """Get currently cached Segmentation"""
        return load_nifti(self.row.segNiftiFilename)

    def get_seg_img_data(self):
        """Get numpy tensor of currently cached segmentation tensor"""
        return get_img_data(self.get_seg_img())

    def get_seg_img_dim(self):
        """
        Get dimension of currently cached Segmentation
        :return: x range, y range, z range (3 return vals)
        """
        return get_dim(self.get_seg_img())




class Segmentations:
    """Iterator for looping over all segmented niftis in the database."""

    def __init__(self, path):
        self.data = NiftiDBHelper(path)
        self.index = 0
        self.total = self.data.q_total()

    def __iter__(self):
        return self

    def next(self):
        if self.index == self.total:
            raise StopIteration
        else:
            self.data.q_x_row(self.index)
            self.index += 1
            return self.data


class SegmentationsShuffled:
    """Iterator for looping over all segmented niftis in the database and giving them back shuffled."""

    def __init__(self, path):
        self.data = NiftiDBHelper(path)
        self.index = 0
        self.total = self.data.q_total()

    def __iter__(self):
        return self

    def next(self):
        if self.index == self.total:
            raise StopIteration
        else:
            self.data.q_shuffled_row()
            self.index += 1
            return self.data


class TestSet:
    def __init__(self, path):
        self.data = NiftiDBHelper(path, type='test')
        self.index = 0
        self.total = self.data.q_total()

    def __iter__(self):
        return self

    def next(self):
        if self.index == self.total:
            raise StopIteration
        else:
            self.data.q_shuffled_row()
            self.index += 1
            return self.data


##
#
# Stateless functions.
#
##

def load_nifti(path):
    """
    Load a nifti by its file path.
    :param path: Path to a nifti file.
    :return: nibabel nifti wrapper obj
    """
    return nib.load(os.path.normpath(path))

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        if len(slices) == 1:
            axes.imshow(slice, cmap="gray")
        else:
            axes[i].imshow(slice, cmap="gray")

def get_dim(img):
    """
    Get dimension of img.
    :param img: Nibabel nifti wrapper.
    :return: x range, y range, z range (3 return vals)
    """
    if len(img.shape) == 3:
        return img.shape[0], img.shape[1], img.shape[2]
    else:
        sys.stderr.write('Error, CT-Image must have 3 dimensions.')
        raise Exception

def get_img_data(img):
    """Get image data of img as numpy tensor"""
    return img.get_data()

def plot_at_center(img):
    """Takes a nibabel nifti wrapper and prints out three centered slices"""
    x, y, z = get_dim(img)

    x_center = round((x - 1) / 2)
    y_center = round((y - 1) / 2)
    z_center = round((z - 1) / 2)

    dat = get_img_data(img)

    show_slices([dat[:, :, z_center], dat[:, y_center, :], dat[x_center, :, :]])
    plt.suptitle("Centered slices for image (xy, xz, yz)")
    plt.show()

def plot_at(x, y, z, img):
    """Takes a nibabel nifti wrapper and prints out three slices a (x, y, z)"""
    x_max, y_max, z_max = get_dim(img)

    if (x > x_max - 1) or (y > y_max - 1) or (z > z_max - 1):
        raise IndexError
    else:
        dat = get_img_data(img)

        show_slices([dat[:, :, z], dat[:, y, :], dat[x, :, :]])
        plt.suptitle("Slices centered at (" + str(x) + ", " + str(y) + ", " + str(z) + "). (xy, xz, yz)")
        plt.show()

def xy_slices(img):
    """Takes a nibabel nifti wrapper and returns a iterator over all xy slices"""
    _, _, z_max = get_dim(img)

    dat = get_img_data(img)

    for i in xrange(z_max - 1):
        yield np.rot90(dat[:, :, i])

def xz_slices(img):
    """Takes a nibabel nifti wrapper and returns a iterator over all xz slices"""
    _, y_max, _ = get_dim(img)

    dat = get_img_data(img)

    for i in xrange(y_max - 1):
        yield np.rot90(dat[:, i, :])

def yz_slices(img):
    """Takes a nibabel nifti wrapper and returns a iterator over all yz slices"""
    x_max, _, _ = get_dim(img)

    dat = get_img_data(img)

    for i in xrange(x_max - 1):
        yield np.rot90(dat[i, :, :])

def serialize_slice(slc):
    """Serialize numpy matrix to byte stream."""
    if np.amax(slc) > 255:
        raise ValueError("Slice must be a uint8 byte array.")
    else:
        return slc.tostring()

def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
    # calc min and max
    min = np.amin(arr)
    max = np.amax(arr)

    arr = np.array(arr, dtype=np.float64)

    if min <= 0:
        # clip to c_min and c_max
        c_arr = np.clip(arr, min * c_min, max * c_max)

        # right shift to zero
        slc_0 = np.add(np.abs(min * c_min), c_arr)
    else:
        # clip to c_max
        c_arr = np.clip(arr, min, max * c_max)

        # left shift to zero
        slc_0 = np.subtract(c_arr, min)

    # normalization
    norm_fac = np.amax(slc_0)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(
                slc_0,
                255
            ),
            np.amax(slc_0)
        )
    else:  # don't divide through 0
        norm = np.multiply(slc_0, 255)

    return norm

def norm_hounsfield_stat(arr, c_min=0.1, c_max=0.3):
    min = np.amin(arr)

    arr = np.array(arr, dtype=np.float64)

    if min <= 0:
        # clip
        c_arr = np.clip(arr, c_min, c_max)

        # right shift to zero
        slc_0 = np.add(np.abs(min), c_arr)
    else:
        # clip
        c_arr = np.clip(arr, c_min, c_max)

        # left shift to zero
        slc_0 = np.subtract(c_arr, min)

    # normalization
    norm_fac = np.amax(slc_0)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(
                slc_0,
                255
            ),
            np.amax(slc_0)
        )
    else:  # don't divide through 0
        norm = np.multiply(slc_0, 255)

    return norm

def hounsfield_to_byte_dyn(arr, c_min=0.1, c_max=0.3):
    """
    Numpy array with hounsfield units will be clipped for contrast enhancement and then normalized to np.uint8.
    Standard clipping is set to favor contrast enhanced tissue. If you want to have a higher bone resolution, set c_max at least to 0.8
    """

    norm = np.clip(np.round(norm_hounsfield_dyn(arr, c_min=c_min, c_max=c_max)), 0, 255)
    return np.array(norm, dtype=np.uint8)

def hounsfield_to_byte_stat(arr, c_min=-150, c_max=300):
    norm = np.clip(np.round(norm_hounsfield_stat(arr, c_min=c_min, c_max=c_max)), 0, 255)
    return np.array(norm, dtype=np.uint8)

def hounsfield_to_float_dyn(arr, c_min=0.1, c_max=0.3):
    norm = np.clip(norm_hounsfield_dyn(arr, c_min=c_min, c_max=c_max), 0, 1)
    return np.array(norm, dtype=np.float64)

def hounsfield_to_float_stat(arr, c_min=-150, c_max=300):
    norm = np.clip(norm_hounsfield_stat(arr, c_min=c_min, c_max=c_max), 0, 1)
    return np.array(norm, dtype=np.float64)

def dump_vol_to_nifti(vol, path):
    vol_trans = None

    # transformation for [0,1] data
    if np.amin(vol) >= 0 and np.amax(vol) <= 1:
        vol_tmp = np.array(vol, dtype=np.float64)
        vol_tmp = np.subtract(vol, 0.5)
        vol_tmp = np.multiply(vol, 2000.0)
        vol_tmp = np.round(vol_tmp)

        vol_trans = np.array(vol_tmp, dtype=np.int16)

    # transformation for data [0,255]
    elif np.amin(vol) >= 0 and np.amax(vol) <= 255:
        vol_tmp = np.array(vol, dtype=np.float64)
        vol_tmp = np.divide(vol_tmp, 255.0)
        vol_tmp = np.subtract(vol, 0.5)
        vol_tmp = np.multiply(vol, 2000.0)
        vol_tmp = np.round(vol_tmp)

        vol_trans = np.array(vol_tmp, dtype=np.int16)

    # transformation for [-1,1] data
    elif np.amin(vol) >= -1 and np.amax(vol) <= 1:
        vol_tmp = np.array(vol, dtype=np.float64)
        vol_tmp = np.multiply(vol, 1000.0)
        vol_tmp = np.round(vol_tmp)

        vol_trans = np.array(vol_tmp, dtype=np.int16)

    # unknown range - just transform to int16
    else:
        vol_tmp = np.round(vol)
        vol_trans = np.array(vol_tmp, dtype=np.int16)

    # simply use identity matrix as affine transformation
    img = nib.Nifti1Image(vol_trans, np.eye(4))

    nib.save(img, path)


def main():
    # ### For test purposes only
    # f3.init_db('/media/nas/sqlitedb/processed/processedFinalOrderedSegmented.sql')
    #
    # #ct_img = nib.load(f3.Segmented.select().first().ctNiftiFilename)
    # ct_img = nib.load('/media/nas/niftis_segmented/CRF_174/2013-12-06/LightSpeed_Pro_32/segmented/368984_liv_x.nii')
    # ct_img_dat = ct_img.get_data()
    #
    # print(ct_img.shape)
    #
    # # Now plot something
    # slice_0 = ct_img_dat[300, :, :]
    # slice_1 = ct_img_dat[:, 300, :]
    # slice_2 = ct_img_dat[:, :, 520]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("Center slices for EPI image")
    # plt.show()
    #
    #
    # ### Close DB
    # f3.close_db()

    # ct_img = nib.load('/media/nas/niftis_segmented/CRF_174/2013-12-06/LightSpeed_Pro_32/segmented/368984.nii')
    # ct_img = nib.load('/media/nas/niftis_segmented/CRF_036/2008-05-09/Emotion_Duo/segmented/03040001.nii')
    # plot_at(150, 350, 550, ct_img)

    nh = NiftiDBHelper('/media/nas/sqlitedb/processed/finalDBNew.sql')
    nh.q_random()
    ct_img = nh.get_ct_img()
    ct_seg = nh.get_seg_img()

    for i, slc in enumerate(xy_slices(ct_img)):

        sol = hounsfield_to_byte_dyn(slc)

        if (i > 1) and (i % 50 == 0):
            # plot seg
            for j, slc_seg in enumerate(xy_slices(ct_seg)):
                if i == j:
                    show_slices([sol, slc_seg])
                    plt.show()
                    break

        print('-')
        print(np.amin(sol))
        print(np.amax(sol))
        print(np.amin(slc))
        print(np.amax(slc))
        #
        # nh = NiftiDBHelper('/media/nas/sqlitedb/processed/processedFinalOrderedSegmented.sql')
        # for i in range(3):
        #     nh.q_random()
        #     plot_at_center(nh.get_ct_img())
        #
        # for dat in Segmentations('/media/nas/sqlitedb/processed/processedFinalOrderedSegmented.sql'):
        #     print(dat.get_ct_img_dim())
        #     print(dat.get_seg_img_dim())

        # print('Hallo Welt!')

        # total = 0
        # for dat in Segmentations('/media/nas/sqlitedb/processed/processedFinalOrderedSegmented.sql'):
        #     print(dat.get_ct_img_dim())
        #     _, _, z = dat.get_ct_img_dim()
        #     total += z
        #
        # print total

    tot_0 = 0.0
    for i in xy_slices(ct_img):
        i = hounsfield_to_byte_dyn(i)

        if np.amax(i) < 5:
            tot_0 += 1

    print('tot blank: ' + str(tot_0))

    ti = 0
    li = 0
    le = 0
    tot = 0
    for s in xy_slices(ct_seg):
        tot +=  1

        s = np.clip(s, 0, 2)

        if np.amax(s) == 2:
            le += 1

        if np.amax(s) == 1:
            li += 1

        if np.amax(s) == 0:
            ti += 1

    print('(total, tissue, liver, lesion)')
    print(str(float(tot)) + ', ' + str(float(float(ti)/float(tot))*100) + ', ' + str(float(float(li)/float(tot))*100) + ', ' + str(float(float(le)/float(tot))*100))



if __name__ == '__main__':
    main()
