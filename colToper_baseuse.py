import os
import torch
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import vgg19, VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# 모델 구조 정의 (학습 때 사용했던 구조와 동일하게)
########################################

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()):
        super(GatedConv2d, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_conv(x)
        m = self.mask_conv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class GatedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=nn.ReLU()):
        super(GatedDeconv2d, self).__init__()
        self.feature_deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_deconv(x)
        m = self.mask_deconv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class ContextualAttention(nn.Module):
    def __init__(self, kernel_size=3, stride=1, dilation=1):
        super(ContextualAttention, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size, stride, dilation, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B,C,H,W = x.size()
        query = x.view(B,C,-1)
        key = x.view(B,C,-1)
        value = x.view(B,C,-1)

        attn = torch.bmm(query.permute(0,2,1), key)
        attn = self.softmax(attn)
        out = torch.bmm(attn, value.permute(0,2,1))
        out = out.permute(0,2,1).view(B,C,H,W)
        out = self.conv(out)
        return out

class InpaintGenerator(nn.Module):
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        self.encoder = nn.Sequential(
            GatedConv2d(4, 64, 4, 2, 1),
            GatedConv2d(64, 128, 4, 2, 1),
            GatedConv2d(128, 256, 4, 2, 1),
            GatedConv2d(256, 512, 4, 2, 1)
        )
        self.contextual_attention = ContextualAttention()
        self.decoder = nn.Sequential(
            GatedDeconv2d(512, 256, 4, 2, 1),
            GatedDeconv2d(256, 128, 4, 2, 1),
            GatedDeconv2d(128, 64, 4, 2, 1),
            GatedDeconv2d(64, 64, 4, 2, 1, activation=nn.ReLU()),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        inp = torch.cat((x, mask), dim=1)
        feat = self.encoder(inp)
        feat = self.contextual_attention(feat)
        out = self.decoder(feat)
        return out

########################################
# 필요하다면 Perceptual Loss 정의 (생략가능)
########################################

# 모델 로드
model_path = "/home/zqrc05/project/imagepro/test/model/colToper/11_best_generator (3).pth"
generator = InpaintGenerator().to(device)
# weights_only=True를 명시하여 FutureWarning 방지
generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
generator.eval()

# 경로 지정
test_images_dir = "/home/zqrc05/project/imagepro/test/output_grayTocol"  # 테스트용 컬러(손상) 이미지 폴더
test_masks_dir = "/home/zqrc05/project/imagepro/test/mask"               # 테스트용 마스크 이미지 폴더
output_dir = "/home/zqrc05/project/imagepro/test/output_colToper_14"      # 결과 저장 폴더

os.makedirs(output_dir, exist_ok=True)

# transform: 학습시와 동일하게
transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor()
])

# 테스트 이미지 폴더 내 모든 이미지에 대해 수행
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(test_images_dir, filename)
        mask_path = os.path.join(test_masks_dir, filename) # 이미지와 동일한 이름의 마스크가 있다고 가정

        if not os.path.exists(mask_path):
            print(f"마스크 이미지 {mask_path} 가 없습니다. 스킵합니다.")
            continue

        # 이미지 로드 (원본 손상 이미지)
        inp_img = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        inp_tensor = transform(inp_img).unsqueeze(0).to(device)   # (1,3,H,W)
        mask_tensor = transform(mask_img).unsqueeze(0).to(device) # (1,1,H,W)

        # 마스크 부분을 0으로 전처리
        mask_broadcast = mask_tensor.expand_as(inp_tensor)  # (1,3,H,W)
        damaged_inp = inp_tensor * (1 - mask_broadcast)

        # 모델 추론
        with torch.no_grad():
            output = generator(damaged_inp, mask_tensor)  # (1,3,H,W)

        # 마스크 영역만 복원 결과 사용, 나머지는 원본 손상 이미지 사용
        # final_result = inp_tensor * (1 - mask) + output * mask
        final_result = inp_tensor * (1 - mask_broadcast) + output * mask_broadcast

        # 결과 텐서를 이미지로 변환
        final_pil = T.ToPILImage()(final_result.squeeze(0).cpu())

        save_path = os.path.join(output_dir, filename)
        final_pil.save(save_path)
        print(f"{filename} 복원 완료(마스크 영역만 덮어쓰기) -> {save_path}")
