from PIL import Image
import os.path
import torch
import torchvision.transforms as transforms

fg_name = ''
mask_name = ''
while True:
    fg_id = input('Please input a foreground id:')
    dirnames = []
    for parent, dirname, filenames in os.walk('OPA/foreground'):
        dirnames.extend(dirname)
    for cat in dirnames:
        fg_name = os.path.join('OPA/foreground/', cat, fg_id + '.jpg')
        mask_name = os.path.join('OPA/foreground/', cat, 'mask_' + fg_id + '.jpg')
        if os.path.exists(fg_name) == True:
            break
    if os.path.exists(fg_name) == False:
        print("This ID does not exist!")
        continue
    else:
        break

bg_name = ''
while True:
    bg_id = input('Please input a background id:')
    dirnames = []
    for parent, dirname, filenames in os.walk('OPA/background'):
        dirnames.extend(dirname)
    for cat in dirnames:
        bg_name = os.path.join('OPA/background/', cat, bg_id + '.jpg')
        if os.path.exists(bg_name) == True:
            break
    if os.path.exists(bg_name) == False:
        print("This ID does not exist!")
        continue
    else:
        break

fg_img = Image.open(fg_name).convert('RGB')
mask_img = Image.open(mask_name).convert('L')
bg_img = Image.open(bg_name).convert('RGB')

bg_h = bg_img.height
bg_w = bg_img.width

print("The size of the background is {} * {}. Please input the position of the foreground.".format(bg_w, bg_h))

str_scale = ''
while True:
    left = int(input('x:'))
    top = int(input('y:'))
    w = int(input('w:'))
    right = w + left
    h = int(input('h:'))
    bottom = h + top

    if right - left <= 0 or bottom - top <= 0 or right > bg_w or bottom > bg_h:
        print('This position is illegal!')
        continue
    else:
        scale = max(w / bg_w, h / bg_h)
        str_scale = "%.9f" % scale
        print("scale=" + str_scale)
        break

fg_transform = transforms.Compose([
    transforms.Resize((bottom - top, right - left)),
    transforms.ToTensor(),
])
fg_img_ = fg_transform(fg_img)
mask_img_ = fg_transform(mask_img)
fg_img = torch.zeros(3, bg_h, bg_w)
mask_img = torch.zeros(3, bg_h, bg_w)
fg_img[:, top:bottom, left:right] = fg_img_[:, :, :]
mask_img[:, top:bottom, left:right] = mask_img_[:, :, :]
bg_img = transforms.ToTensor()(bg_img)
blended = fg_img * mask_img + bg_img * (1 - mask_img)
com_pic = transforms.ToPILImage()(blended).convert('RGB')
# com_pic.show()

while True:
    label = input("Please input a label:")
    if label != '0' and label != '1':
        print('This label is illegal!')
        continue
    else:
        break

while True:
    save_path = input('Please input a path to save your composite image:')
    if os.path.exists(save_path) == False:
        print('This path does not exist!')
        continue
    else:
        break
com_pic_name = fg_id + "_" + bg_id + "_" + str(left) + "_" + str(top) + "_" + str(w) + "_" + str(
    h) + "_" + '%.4f' % eval(str_scale) + "_" + label + ".jpg"
save_path = os.path.join(save_path, com_pic_name)
com_pic.save(save_path)
