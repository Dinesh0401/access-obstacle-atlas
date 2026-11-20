"""Draw an illustrative crossing and normalized evidence boxes for the review desk."""

from PIL import Image, ImageDraw, ImageFont


def example_crossing():
    image = Image.new("RGB", (960, 530), "#cadbd8")
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, 140), (960, 124), (960, 288), (0, 350)], fill="#d9c9b4")
    for x, height, width in [
        (0, 146, 170),
        (180, 88, 154),
        (345, 126, 200),
        (564, 99, 115),
        (690, 148, 188),
        (899, 130, 100),
    ]:
        draw.rectangle((x, 161 - height, x + width, 170), fill="#8fa4a1")
        for wx in range(x + 14, x + width - 10, 29):
            for wy in range(170 - height, 151, 33):
                draw.rectangle((wx, wy, wx + 13, wy + 18), fill="#657b7c")
    draw.polygon([(0, 333), (960, 241), (960, 530), (0, 530)], fill="#43545b")
    draw.line([(0, 331), (960, 240)], fill="#f6e8cf", width=9)
    for x in range(-70, 960, 132):
        draw.polygon(
            [(x + 124, 365), (x + 193, 362), (x + 83, 530), (x - 28, 530)],
            fill="#cdd5cd",
        )
    draw.polygon([(347, 269), (504, 255), (581, 279), (318, 304)], fill="#e3bd7b")
    for i in range(7):
        draw.line([(355 + i * 21, 272), (341 + i * 31, 297)], fill="#b89055", width=3)
    for x, y in [(381, 252), (492, 244)]:
        draw.ellipse((x - 21, y - 21, x + 21, y + 21), fill="#26343a")
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#aab6ae")
    draw.polygon(
        [
            (365, 243),
            (383, 218),
            (408, 215),
            (428, 237),
            (466, 233),
            (480, 194),
            (490, 196),
            (504, 238),
            (481, 240),
            (460, 247),
            (395, 253),
        ],
        fill="#b45e45",
    )
    draw.line([(475, 195), (467, 176), (486, 174)], fill="#26343a", width=5)
    draw.rounded_rectangle((377, 210, 413, 220), radius=5, fill="#26343a")
    return image


def evidence_image(image, marks):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for index, (label, box) in enumerate(marks, 1):
        left, top, right, bottom = [
            int(v * length) for v, length in zip(box, (width, height, width, height))
        ]
        stroke = max(2, width // 260)
        color = "#ffd066"
        draw.rectangle((left, top, right, bottom), outline=color, width=stroke)
        font = ImageFont.load_default(size=max(12, width // 52))
        title = f"{index}  {label}"
        x, y = left, max(0, top - width // 35)
        bbox = draw.textbbox((x + 4, y + 2), title, font=font)
        draw.rectangle((x, y, bbox[2] + 5, bbox[3] + 3), fill=color)
        draw.text((x + 4, y + 2), title, fill="#18242a", font=font)
    return image
