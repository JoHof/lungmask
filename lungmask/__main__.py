import argparse
import os
import sys
from importlib import metadata

import numpy as np
import SimpleITK as sitk

from lungmask import LMInferer, utils
from lungmask.logger import logger


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f"File not found: {string}")


def main():
    version = metadata.version("lungmask")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=path,
        help="Path to the input image, can be a folder for dicoms",
    )
    parser.add_argument(
        "output", metavar="output", type=str, help="Filepath for output lungmask"
    )
    parser.add_argument(
        "--modelname",
        help="spcifies the trained model, Default: R231",
        type=str,
        choices=["R231", "LTRCLobes", "LTRCLobes_R231", "R231CovidWeb"],
        default="R231",
    )
    parser.add_argument(
        "--modelpath", help="spcifies the path to the trained model", default=None
    )
    parser.add_argument(
        "--cpu",
        help="Force using the CPU even when a GPU is available, will override batchsize to 1",
        action="store_true",
    )
    parser.add_argument(
        "--nopostprocess",
        help="Deactivates postprocessing (removal of unconnected components and hole filling)",
        action="store_true",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.",
        default=20,
    )
    parser.add_argument(
        "--noprogress",
        action="store_true",
        help="If set, no tqdm progress bar will be shown",
    )
    parser.add_argument(
        "--version",
        help="Shows the current version of lungmask",
        action="version",
        version=version,
    )
    parser.add_argument(
        "--removemetadata",
        action="store_true",
        help="Do not keep study/patient related metadata of the input, if any.",
    )

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    # keeping any Patient / Study info is the default, deactivate in case of arg specified or non-HU data
    keeppatinfo = not args.removemetadata

    logger.info("Load model")

    input_image = utils.load_input_image(
        args.input, disable_tqdm=args.noprogress, read_metadata=keeppatinfo
    )

    logger.info("Infer lungmask")
    if args.modelname == "LTRCLobes_R231":
        assert (
            args.modelpath is None
        ), "Modelpath can not be specified for LTRCLobes_R231 mode"
        inferer = LMInferer(
            modelname="LTRCLobes",
            force_cpu=args.cpu,
            fillmodel="R231",
            batch_size=batchsize,
            volume_postprocessing=not (args.nopostprocess),
            tqdm_disable=args.noprogress,
        )
        result = inferer.apply(input_image)
    else:
        inferer = LMInferer(
            modelname=args.modelname,
            modelpath=args.modelpath,
            force_cpu=args.cpu,
            batch_size=batchsize,
            volume_postprocessing=not (args.nopostprocess),
            tqdm_disable=args.noprogress,
        )
        result = inferer.apply(input_image)

    result_out = sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.output)

    if keeppatinfo:
        # keep the Study Instance UID
        writer.SetKeepOriginalImageUID(True)

        DICOM_tags_to_keep = utils.get_DICOM_tags_to_keep()

        # copy the DICOM tags we want to keep
        for key in input_image.GetMetaDataKeys():
            if key in DICOM_tags_to_keep:
                result_out.SetMetaData(key, input_image.GetMetaData(key))

        # set the Series Description tag
        result_out.SetMetaData("0008|103e", "Created with lungmask")

        # set WL/WW
        result_out.SetMetaData("0028|1050", "1")  # Window Center
        result_out.SetMetaData("0028|1051", "2")  # Window Width

    logger.info(f"Save result to: {args.output}")
    writer.Execute(result_out)


if __name__ == "__main__":
    print("called as script")
    main()
