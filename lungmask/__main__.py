import sys
import argparse
import logging
from lungmask import mask
from lungmask import utils
import os
import SimpleITK as sitk


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='input', type=path, help='Path to the input image, can be a folder for dicoms')
    parser.add_argument('output', metavar='output', type=str, help='Filepath for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet', 'resunet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str, choices=['R231','LTRCLobes','R231CovidWeb'], default='R231')
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1", action='store_true')
    parser.add_argument('--nopostprocess', help="Deactivates postprocessing (removal of unconnected components and hole filling", action='store_true')
    parser.add_argument('--batchsize', type=int, help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.", default=20)

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    logging.info(f'Load model')
    model = mask.get_model(args.modeltype, args.modelname)
    input_image = utils.get_input_image(args.input)
    logging.info(f'Infer lungmask')
    result = mask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize, volume_postprocessing=not(args.nopostprocess))

    result_out= sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)
    logging.info(f'Save result to: {args.output}')
    sys.exit(sitk.WriteImage(result_out, args.output))


if __name__ == "__main__":
    print('called as script')
    main()
