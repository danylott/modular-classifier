from PIL import Image, ImageDraw


def label_image(path_to_image, path_to_save, annotation):
    with Image.open(path_to_image).convert('RGBA') as base:
        # base.size = (WIDTH, HEIGHT)
        width = base.size[0]
        height = base.size[1]
        annotation = [int(coord * base.size[idx % 2]) for idx, coord in enumerate(annotation)]

        image = Image.new('RGBA', base.size, (255, 255, 255, 0))

        d = ImageDraw.Draw(image)
        d.polygon(annotation, fill=(255, 0, 0, 128))

        out = Image.alpha_composite(base, image)
        out.save(path_to_save)

    return annotation, width, height


if __name__ == "__main__":  
    label_image('../../dataset/front_sided_images/IMG_0430r.JPG', 'test.png', [0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5])
