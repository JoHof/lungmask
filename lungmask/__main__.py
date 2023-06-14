import argparse
import logging
import os
import sys

import numpy as np
import pkg_resources  # type: ignore
import SimpleITK as sitk

from lungmask import LMInferer, utils


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f"File not found: {string}")


def main():
    version = pkg_resources.require("lungmask")[0].version

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
        "--modeltype", help="Default: unet", type=str, choices=["unet"], default="unet"
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
        "--classes",
        help="spcifies the number of output classes of the model",
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
        "--noHU",
        help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images",
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

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    if args.classes is not None:
        logging.warn(
            "!!! Warning: The `classes` parameter is deprecated and will be removed in the next version !!!"
        )

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    logging.info("Load model")

    input_image = utils.load_input_image(args.input, disable_tqdm=args.noprogress)
    logging.info("Infer lungmask")
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
            noHU=args.noHU,
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
            noHU=args.noHU,
            tqdm_disable=args.noprogress,
        )
        result = inferer.apply(input_image)

    if args.noHU:
        file_ending = args.output.split(".")[-1]
        if file_ending in ["jpg", "jpeg", "png"]:
            result = (result / (result.max()) * 255).astype(np.uint8)
        result = result[0]

    result_out = sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)
    logging.info(f"Save result to: {args.output}")
    sitk.WriteImage(result_out, args.output)


if __name__ == "__main__":
    print("called as script")
    main()
