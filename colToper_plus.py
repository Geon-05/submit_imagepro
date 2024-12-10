import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################
# 모델 구조 정의 (학습시 사용한 것과 동일)
############################

class GatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=torch.nn.ReLU()):
        super(GatedConv2d, self).__init__()
        self.feature_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_conv(x)
        m = self.mask_conv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class GatedDeconv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=torch.nn.ReLU()):
        super(GatedDeconv2d, self).__init__()
        self.feature_deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        f = self.feature_deconv(x)
        m = self.mask_deconv(x)
        gated = self.sigmoid(m)
        if self.activation is not None:
            f = self.activation(f)
        return f * gated

class ContextualAttention(torch.nn.Module):
    def __init__(self, kernel_size=3, stride=1, dilation=1):
        super(ContextualAttention, self).__init__()
        self.conv = torch.nn.Conv2d(512, 512, kernel_size, stride, dilation, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

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

class Stage1Generator(torch.nn.Module):
    def __init__(self):
        super(Stage1Generator, self).__init__()
        self.encoder = torch.nn.Sequential(
            GatedConv2d(4, 64, 4, 2, 1),
            GatedConv2d(64, 128, 4, 2, 1),
            GatedConv2d(128, 256, 4, 2, 1),
            GatedConv2d(256, 512, 4, 2, 1)
        )
        self.decoder = torch.nn.Sequential(
            GatedDeconv2d(512, 256, 4, 2, 1),
            GatedDeconv2d(256, 128, 4, 2, 1),
            GatedDeconv2d(128, 64, 4, 2, 1),
            GatedDeconv2d(64, 64, 4, 2, 1, activation=torch.nn.ReLU()),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, mask):
        inp = torch.cat((x, mask), dim=1)
        feat = self.encoder(inp)
        out = self.decoder(feat)
        return out

class Stage2Generator(torch.nn.Module):
    def __init__(self):
        super(Stage2Generator, self).__init__()
        self.encoder = torch.nn.Sequential(
            GatedConv2d(7, 64, 4, 2, 1),
            GatedConv2d(64, 128, 4, 2, 1),
            GatedConv2d(128, 256, 4, 2, 1),
            GatedConv2d(256, 512, 4, 2, 1)
        )
        self.contextual_attention = ContextualAttention()
        self.decoder = torch.nn.Sequential(
            GatedDeconv2d(512, 256, 4, 2, 1),
            GatedDeconv2d(256, 128, 4, 2, 1),
            GatedDeconv2d(128, 64, 4, 2, 1),
            GatedDeconv2d(64, 64, 4, 2, 1, activation=torch.nn.ReLU()),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_out, inp, mask):
        fin_inp = torch.cat((coarse_out, inp, mask), dim=1)
        feat = self.encoder(fin_inp)
        feat = self.contextual_attention(feat)
        out = self.decoder(feat)
        return out

############################
# 테스트용 코드
############################

# 최적 모델 가중치 경로 (학습 완료 후 저장한 best 모델 경로 지정)
best_coarse_path = "/home/zqrc05/project/imagepro/test/model/colToperPlus/12_best_coarse_generator_epoch9.pth"
best_fine_path = "/home/zqrc05/project/imagepro/test/model/colToperPlus/12_best_fine_generator_epoch9.pth"

# 테스트 이미지(손상된 컬러 이미지) 폴더
test_input_dir = "/home/zqrc05/project/imagepro/test/output_grayTocol"  # 손상된 컬러이미지 폴더 경로
test_mask_dir = "/home/zqrc05/project/imagepro/test/mask"          # 마스크 이미지 폴더 경로
output_dir = "/home/zqrc05/project/imagepro/test/output_colToper_12"          # 결과 저장 폴더 경로

os.makedirs(output_dir, exist_ok=True)

# 모델 로드
coarse_generator = Stage1Generator().to(device)
fine_generator = Stage2Generator().to(device)

# weights_only=True를 명시하여 FutureWarning 방지
coarse_generator.load_state_dict(torch.load(best_coarse_path, map_location=device, weights_only=True))
fine_generator.load_state_dict(torch.load(best_fine_path, map_location=device, weights_only=True))

coarse_generator.eval()
fine_generator.eval()

# transform: 학습시와 동일하게
transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor()
])

# 테스트 이미지 리스트 수집
test_files = [f for f in os.listdir(test_input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

with torch.no_grad():
    for filename in test_files:
        input_path = os.path.join(test_input_dir, filename)
        mask_path = os.path.join(test_mask_dir, filename)  # 마스크도 같은 이름으로 존재한다고 가정

        if not os.path.exists(mask_path):
            print(f"{mask_path}가 존재하지 않습니다. 스킵합니다.")
            continue

        # 이미지 로드
        inp_img = Image.open(input_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        inp_tensor = transform(inp_img).unsqueeze(0).to(device)   # (1,3,H,W)
        mask_tensor = transform(mask_img).unsqueeze(0).to(device) # (1,1,H,W)

        # 마스크 영역을 0으로 전처리 (손상부)
        mask_broadcast = mask_tensor.expand_as(inp_tensor)
        inp_tensor = inp_tensor * (1 - mask_broadcast)

        # 추론(코스 단계)
        coarse_out = coarse_generator(inp_tensor, mask_tensor)
        # 추론(파인 단계)
        fine_out = fine_generator(coarse_out, inp_tensor, mask_tensor)

        # 결과 저장
        result = fine_out.squeeze(0).cpu()  # (3,H,W)
        result_pil = T.ToPILImage()(result)

        save_path = os.path.join(output_dir, filename)
        result_pil.save(save_path)
        print(f"{filename} 복원 완료 -> {save_path}")
