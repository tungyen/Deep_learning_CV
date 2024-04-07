from PIL import Image

def imageResize(path, size=(256, 256)):
    img = Image.open(path)
    longest = max(img.size)
    mask = Image.new('RGB', (longest, longest), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def maskResize(path, size=(256, 256)):
    img = Image.open(path).convert('P')
    longest = max(img.size)
    mask = Image.new('P', (longest, longest), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask