from PIL import Image
import SimpleITK as sitk
from preProcessor_v2 import preProcessScan

ct = 'dataset/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
mask = 'dataset/seg_lungs/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'

ct = sitk.ReadImage(ct)
ctArr = sitk.GetArrayFromImage(ct)

mask = sitk.ReadImage(mask)
maskArr = sitk.GetArrayFromImage(mask)

processed_scan = preProcessScan(ct[80], mask[80])
print(processed_scan)
pilImg = Image.fromarray(processed_scan)

pilImg = Image.fromarray