import json
import argparse
import funcy
from os import path
# from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: i["id"], images)
    return funcy.lfilter(lambda a: a["image_id"] in image_ids, annotations)


def split_coco(args):
    with open(args.annotations, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        images_with_annotations = funcy.lmap(lambda a: a["image_id"], annotations)

        images = funcy.lremove(lambda i: i["id"] not in images_with_annotations, images)
        images = funcy.select(
            lambda a: path.exists(path.join(args.dataset, a["file_name"])), images
        )

        annotation_map = funcy.group_by(lambda i: i["category_id"], annotations)

        test_ids = []
        for i in annotation_map.values():
            test_ids = test_ids + funcy.lmap(lambda j: j["image_id"], i[:max(args.mintest, int(len(i) * args.split))])

        test, train = funcy.lsplit(lambda i: i["id"] in test_ids, images)
        # x, y = train_test_split(images, train_size=args.split)

        save_coco(
            args.train,
            info,
            licenses,
            train,
            filter_annotations(annotations, train),
            categories,
        )
        save_coco(
            args.test,
            info,
            licenses,
            test,
            filter_annotations(annotations, test),
            categories,
        )

        print(
            "Saved {} entries in {} and {} in {}".format(
                len(train), args.train, len(test), args.test
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits COCO annotations file into training and test sets."
    )
    parser.add_argument(
        "annotations",
        metavar="coco_annotations",
        type=str,
        help="Path to COCO annotations file.",
    )
    parser.add_argument("train", type=str, help="Where to store COCO training annotations")
    parser.add_argument("test", type=str, help="Where to store COCO test annotations")
    parser.add_argument("dataset", type=str, help="Where are images located")
    parser.add_argument("mintest", type=int, help="Min count of images to test model")
    parser.add_argument(
        "-s",
        dest="split",
        type=float,
        required=True,
        help="A percentage of a split; a number in (0, 1)",
    )

    args = parser.parse_args()
    split_coco(args)
