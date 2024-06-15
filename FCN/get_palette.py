import json
import numpy as np
from PIL import Image


target = Image.open("./2007_001288.png")
palette = target.getpalette()
palette = np.reshape(palette, (-1, 3)).tolist()
pd = dict((i, color) for i, color in enumerate(palette))

json_str = json.dumps(pd)
with open("palette.json", "w") as f:
    f.write(json_str)
