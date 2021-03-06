#!/usr/bin/env python3
"""
Uses iNaturalist vision model to search for birds in frame,
takes a picture when a bird is found and writes identification to a text file

TODOs:
fix .service file
"""
import time
import argparse
import contextlib

from aiy.board import Board

from aiy.vision.inference import CameraInference
from aiy.vision.models import inaturalist_classification

from picamera import PiCamera

boring = ["Meleagris gallopavo (Wild Turkey)", "background"]
# because apparently the bird feeder looks like a turkey


def write_observation(obs):
    """
    writes a timestamped observation to a .txt file
    """
    with open('birdimages/birdlist.txt', 'a') as birdlist:
        birdlist.write(obs)


def lookforbirds(model_type, thresh, topk):
    """
    Main driver code for inference and camera capture
    """

    with contextlib.ExitStack() as stack:
        board = stack.enter_context(Board())
        camera = stack.enter_context(PiCamera(sensor_mode=4, framerate=30))

        # Configure camera
        camera.resolution = (1640, 922)
        camera.awb_mode = 'auto' # auto white balance
        camera.start_preview()

        # Capture a sample ROI image
        camera.capture("birdimages/ROI_sample.jpg")
        print("Running Bird Trigger Camera")

        time.sleep(2)

        # Inference
        with CameraInference(inaturalist_classification.model(model_type)) as inference:

            for result in inference.run():

                if len(inaturalist_classification.get_classes(result)) >= 1:

                    classes = inaturalist_classification.get_classes(result,
                                                             top_k=topk,
                                                             threshold=thresh)

                    for i, (label, score) in enumerate(classes):

                        if(label not in boring): # filtering camera artifacts
                            print('Result %d: %s (prob=%f)' % (i, label, score))

                            # write observation to file
                            obs = time.strftime("%y-%m-%d_%H-%M-%S") + " %s (prob=%f) \n" % (label, score)
                            write_observation(obs)

                            # get species info
                            id = label.split(' ')

                            # save picture
                            camera.annotate_text = id[0]+' '+id[1]+' (prob= %.2f)' % score
                            camera.capture("birdimages/"+time.strftime("%y%m%d_%H%M%S_")+id[0]+"_"+id[1]+".jpg")
                            time.sleep(20)


        camera.stop_preview()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                        help='Classification probability threshold.')

    parser.add_argument('--top_k', '-n', type=int, default=1,
                        help='Max number of returned classes.')

    parser.add_argument('--model', '-m', choices=('plants', 'insects', 'birds'),
                        required=False, default='birds', help='Model to run.')

    args = parser.parse_args()


    model_type = {'plants':  inaturalist_classification.PLANTS,
                  'insects': inaturalist_classification.INSECTS,
                  'birds':   inaturalist_classification.BIRDS}[args.model]

    lookforbirds(model_type, args.threshold, args.top_k)
