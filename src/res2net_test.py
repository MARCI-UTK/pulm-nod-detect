import urllib3
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

fname = 'dataset/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
rawItkImg = sitk.