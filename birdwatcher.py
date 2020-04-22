#!/usr/bin/env python3
"""
Uses iNaturalist vision model to search for birds in frame,
takes picture when bird is found and writes identification to a text file

TODOs: annotate images, include iNaturalist 3rd-party app integration
"""
import time
import argparse
import contextlib

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.toneplayer import TonePlayer

from aiy.vision.inference import CameraInference
from aiy.vision.models import inaturalist_classification

from picamera import PiCamera

boring = ["Meleagris gallopavo (Wild Turkey)", "background"]
# because apparently the bird feeder looks like a turkey

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                        help='Classification probability threshold.')
    parser.add_argument('--top_k', '-n', type=int, default=1,
                        help='Max number of returned classes.')
    parser.add_argument('--sparse', '-s', action='store_true', default=False,
                        help='Use sparse tensors.')
    parser.add_argument('--model', '-m', choices=('plants', 'insects', 'birds'),
                        required=False, default='birds', help='Model to run.')
    args = parser.parse_args()


    model_type = {'plants':  inaturalist_classification.PLANTS,
                  'insects': inaturalist_classification.INSECTS,
                  'birds':   inaturalist_classification.BIRDS}[args.model]

    with contextlib.ExitStack() as stack:
        leds = stack.enter_context(Leds())
        board = stack.enter_context(Board())

        camera = stack.enter_context(PiCamera())

        # Configure camera
        camera.resolution = (1640, 922)
        #camera.exposure_mode = 'sports'
        camera.crop = (0.33, 0.45, 0.3, 0.3)
        camera.start_preview()

        print("Running Bird Trigger Camera")

        camera.capture("birdimages/ROI_sample.jpg")
        time.sleep(2)

        # Inference
        with CameraInference(inaturalist_classification.model(model_type)) as inference:

            for result in inference.run():

                if len(inaturalist_classification.get_classes(result)) >= 1:
                    
                    classes = inaturalist_classification.get_classes(result,
                                                             top_k=args.top_k,
                                                             threshold=args.threshold)


                    for i, (label, score) in enumerate(classes):

                        if(label not in boring):
                            print('Result %d: %s (prob=%f)' % (i, label, score))

                            birdlist = open("birdimages/birdlist.txt","a") #append mode
                            birdlist.write(time.strftime("%y-%m-%d_%H-%M-%S")+" %s (prob=%f) \n" % (label, score))
                            birdlist.close()

                            camera.capture("birdimages/"+time.strftime("%y-%m-%d_%H-%M-%S")+".jpg")
                        else:
                            print("Ignoring image artifact: "+label)
                    # TODO: fix timing and when pictures are taken
                    time.sleep(30)


        camera.stop_preview()


if __name__ == '__main__':
    main()
