#!/usr/bin/env python3

from pathlib import Path

from matplotlib import image

from fc_ground_truth import FCGroundTruth


class FCFingerprint:
    def __init__(self, fp_path, fp_fname):
        self.fp_path = fp_path
        self.fp_fname = fp_fname

        img_path = Path(fp_path).joinpath(fp_fname + '.png')
        self.image = image.imread(img_path)

        gt_path = Path(fp_path).joinpath(fp_fname + '.txt')
        self.gt = FCGroundTruth.from_file(gt_path)
        # TODO: Check if real or synthetic from the path
        self.fp_type = 'TODO'

    def __repr__(self):
        return 'FCFingerprint({}, {})'.format(self.fp_path, self.fp_fname)

    def __str__(self):
        h, w = self.image.shape
        return '{}x{} {} {} FCFingerprint filename {}'.format(w, h,
                                                              self.fp_type,
                                                              self.gt.fp_class,
                                                              self.fp_fname)


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-p', '--fp-path', dest='fp_path',
                        default='datasets/fc/png_txt/figs_0',
                        metavar='BASEPATH',
                        help='directory for fingerprint files')
    parser.add_argument('-fn', '--fp-fname', dest='fp_fname',
                        default='f0001_01', metavar='FINGERPRINT FILE NAME',
                        help='name of the fingerprint file')

    args = parser.parse_args(sys.argv[1:])

    fp_path = args.fp_path
    fp_fname = args.fp_fname

    fp = FCFingerprint(fp_path, fp_fname)

    print('Created a {} fingerprint with id {}'.format(fp.fp_type,
                                                       fp.fp_fname))
    print(fp)
