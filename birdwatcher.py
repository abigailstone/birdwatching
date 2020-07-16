#!/usr/bin/env python3
"""
Uses iNaturalist vision model to search for birds in frame,
takes a picture when a bird is found and writes identification to a text file

TODOs:
switch prints to logger
fix .service file
"""
import time
import argparse
import contextlib

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed

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
        camera.awb_mode = 'sunlight'
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
                                                             top_k=args.top_k,
                                                             threshold=args.threshold)

                    for i, (label, score) in enumerate(classes):

                        if(label not in boring):
                            print('Result %d: %s (prob=%f)' % (i, label, score))

                            # write observation to file
                            birdlist = open("birdimages/birdlist.txt","a") #append mode
                            birdlist.write(time.strftime("%y-%m-%d_%H-%M-%S")+" %s (prob=%f) \n" % (label, score))
                            birdlist.close()

                            id = label.split(' ')

                            # save picture
                            camera.annotate_text(id[0]+' '+id[1]+' (prob='+int(score*100)+'%)')
                            camera.capture("birdimages/"+time.strftime("%y%m%d_%H%M%S_")+id[0]+"_"+id[1]+".jpg")
                            time.sleep(20)
                            

        camera.stop_preview()


if __name__ == '__main__':
    main()
