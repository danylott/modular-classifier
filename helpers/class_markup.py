from PIL import Image, ImageDraw


def class_markup(path_to_image, path_to_save, markups):
    with Image.open(path_to_image).convert('RGBA') as base:
        for markup in markups:
            image = Image.new('RGBA', base.size, (255, 255, 255, 0))
            # base.size = (WIDTH, HEIGHT)
            shape = [(markup['x'] * base.size[0], markup['y'] * base.size[1]),
                     (markup['x'] * base.size[0] + markup['w'] * base.size[0],
                      markup['y'] * base.size[1] + markup['h'] * base.size[1])]

            d = ImageDraw.Draw(image)
            d.rectangle(shape, fill=(255, 0, 0, 128))

            base = Image.alpha_composite(base, image)
        base.save(path_to_save)

    return True


if __name__ == "__main__":
    class_markup('../../dataset/front_sided_images/IMG_0430r.JPG', 'test2.png', [
        {
          "field": "Model",
          "x": 0.57,
          "y": 0.41,
          "w": 0.22,
          "h": 0.13
        },
        {
          "field": "Size",
          "x": 0.29,
          "y": 0.51,
          "w": 0.1,
          "h": 0.17
        },
        {
          "field": "Color",
          "x": 0.32,
          "y": 0.13,
          "w": 0.25,
          "h": 0.16
        }
      ])
