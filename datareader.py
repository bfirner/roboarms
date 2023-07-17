#!/usr/bin/env python3

# Read some data from a rosbag
# Recording was probably done with something like this:
# > ros2 bag record /arm1/joint_states /arm2/joint_states /image_raw/compressed /arm1/robot_description /arm2/robot_description /camera_info

# arm1 is the manipulator and arm2 is the pupper. Images should be suitable for DNN training.

import argparse
import cv2
import numpy
import sys
import yaml

from nml_bag import Reader


def isKey(cvkey, key):
    # OpenCV's waitKey function returns not just the key, but also any keyboard modifiers. This
    # means that the returned value cannot be compared to just the key.
    return key == (cvkey & 0xFF)

arrow_left  = 81
arrow_up    = 82
arrow_right = 83
arrow_down  = 84


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_robot', default='arm2')
    p.add_argument('--bag_path', required=True, type=str)
    # TODO Use. Determines if all data should be preloaded or read sequentially. Speeds up labelling
    # operations.
    p.add_argument('--preload', required=False, default=False)
    # Path to the label file.
    p.add_argument('--label_file', default=None)

    args = p.parse_args()

    if args.label_file is None:
        args.label_file = ''.join(args.bag_path.split('.')[:-1]) + ".yaml"

    print(f"Labels will be saved to {args.label_file}")

    arm_topic = f"/{args.train_robot}/joint_states"
    image_topic = "/image_raw/compressed"
    camera_info = "/camera_info"
    reader = Reader(filepath=args.bag_path, topics=[arm_topic, image_topic, camera_info])

    # Print the topics.
    print(f'The bag contains the following topics:')
    print(reader.topics)
    
    # Print the mapping between topics and message types.
    print(f'The message types associated with each topic are as follows:')
    print(reader.type_map)

    # The entire ros2bag can be fetched with reader.records(), which preloads and caches the result.
    # This is the preferred method of labelling, so long as we can guarantee that enough memory will
    # always be available.
    
    # Print the messages stored in the bag.
    #print(f'All message data records:')
    #print(reader.records)
    record_idx = 0
    if not args.preload:
        I = reader.__iter__()
    finished = False
    while not finished:
        # Advance to the next record
        if args.preload:
            if record_idx + 1 < len(reader.records()):
                record_idx += 1
                record = reader.records()[record_idx]
        else:
            try:
                record = I.__next__()
                record_idx += 1
            except StopIteration:
                finished = True
        print(f"record {record_idx} has keys {list(record.keys())}")
        print(record['topic'])
        if record['topic'] == arm_topic:
            print('\t', record['position'])
            print('\t', record['velocity'])
            print('\t', record['effort'])
        if record['topic'] == image_topic:
            print('\t', record['format'])
            bytestr = bytes(record['data'])
            np_arr = numpy.frombuffer(bytestr, numpy.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            # Pause on the displayed window until a keystroke is received
            cv2.imshow(image_topic, img)

            # Keep getting input until the image changes or labelling is finished
            next_record = record_idx
            while not finished and next_record == record_idx:
                inkey = cv2.waitKey(0)
                # Finish if the user inputs q
                finished = isKey(inkey, ord('q'))
                if isKey(inkey, arrow_right):
                    next_record += 1
                elif isKey(inkey, arrow_left):
                    next_record -= 1
                # TODO Use the returned key value for labelling.
                # Save the per-frame segment labels into the file located at args.label_file
                # The dataprep task will need to read in both that yaml file and the ros2bag.
            if args.preload:
                # TODO FIXME: Actually needs to seek to the next or previous image, not just any
                # old record
                record_idx = next_record-1
            # TODO FIXME Print some stuff for the UI:
            #   cv2.putText( img, text, (y, x), cv2.FONT_HERSHEY_SIMPLEX, 1, (r,g,b), thickness, lineType, bottomLeftOrigin(bool))

        if record['topic'] == camera_info:
            print('\t', record['height'])
            print('\t', record['width'])

    # Remove the window
    cv2.destroyAllWindows()
    
    # Return
    return



if __name__ == '__main__':
    main()
