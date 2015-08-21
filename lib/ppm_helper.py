__author__ = 'Marc'

import os
import io


class PPM(object):

    @staticmethod
    def write(img, path, filename):
        with io.open(os.path.normpath(path + '/' + filename + '.ppm'), 'wb') as ppm:
            ppm.write(b'P5\n' + bytes(img.shape[1]) + b' ' + bytes(img.shape[0]) + b'\n' + b'255\n')
            ppm.write(img.tostring())