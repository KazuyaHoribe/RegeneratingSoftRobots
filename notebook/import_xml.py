import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


def read_vxa(fname):
    tree = ET.parse(fname)
    root: Element = tree.getroot()
    # root > VXC > Structure > Data > Layer
    layers = root.find('VXC').find('Structure').find('Data').findall('Layer')
    layers = [layer.text for layer in layers]
    layers = [[list(map(int, list(layer[i:i + 7]))) for i in range(0,len(layer), 7)] for layer in layers]
    print(layers)
    return layers


if __name__ == '__main__':
    import numpy as np

    mats = read_vxa('biped.vxa')
    print(np.array(mats).shape )