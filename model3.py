import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Các khối xây dựng ---

class DepthwiseSeparableConv(nn.Module):
    """
    Tích chập tách biệt theo chiều sâu (Depthwise Separable Convolution).
    Bao gồm: Conv chiều sâu (depthwise) -> BatchNorm -> ReLU -> Conv điểm (pointwise) -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class AsymmetricDepthwiseSeparableConv(nn.Module):
    """
    Tích chập tách biệt theo chiều sâu bất đối xứng (Asymmetric Depthwise Separable Convolution).
    Một biến thể để giảm tham số hơn nữa.
    Bao gồm: Conv chiều sâu 1xK -> Conv chiều sâu Kx1 -> BatchNorm -> ReLU -> Conv điểm (pointwise) -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AsymmetricDepthwiseSeparableConv, self).__init__()
        # Tách kernel 3x3 thành 1x3 và 3x1
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),
                                    stride=(1, stride), padding=(0, padding), groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1),
                                    stride=(stride, 1), padding=(padding, 0), groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise1(x)
        x = self.depthwise2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DecoderBlockWithSkip(nn.Module):
    """
    Khối giải mã có kết nối tắt (Skip Connection).
    Upsampling -> Concatenate(Upsampled, Skip_Feature) -> Conv 1x1 -> BN -> ReLU
    """
    def __init__(self, skip_channels, upsampled_channels, out_channels):
        super(DecoderBlockWithSkip, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Số kênh đầu vào cho Conv 1x1 là tổng số kênh từ skip connection và upsampled feature
        total_in_channels = skip_channels + upsampled_channels
        self.conv = nn.Conv2d(total_in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_feature):
        """
        Args:
            x (torch.Tensor): Tensor đầu vào từ lớp decoder trước đó.
            skip_feature (torch.Tensor): Tensor từ kết nối tắt của encoder.
        """
        x = self.upsample(x)
        # Nối (concatenate) theo chiều kênh (dimension 1)
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# --- Kiến trúc TinySegNet với Skip Connections ---

class TinySegNetWithSkip(nn.Module):
    """
    Kiến trúc TinySegNet được cập nhật với Skip Connections.
    """
    def __init__(self, num_classes, input_channels=3):
        super(TinySegNetWithSkip, self).__init__()

        # --- Bộ mã hóa (Encoder) ---
        # Lưu trữ số kênh đầu ra của các tầng encoder để dùng cho skip connections
        self.enc_channels = [16, 32, 64, 128]

        # Lớp đầu vào
        self.entry_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.enc_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.enc_channels[0]),
            nn.ReLU(inplace=True)
        ) # Output: H/2, W/2, C=16

        # Các khối mã hóa
        self.enc_block1 = DepthwiseSeparableConv(self.enc_channels[0], self.enc_channels[1], stride=2) # Output: H/4, W/4, C=32
        self.enc_block2 = AsymmetricDepthwiseSeparableConv(self.enc_channels[1], self.enc_channels[2], stride=2) # Output: H/8, W/8, C=64
        self.enc_block3 = DepthwiseSeparableConv(self.enc_channels[2], self.enc_channels[3], stride=2) # Output: H/16, W/16, C=128
        # Bottleneck (có thể dùng Asymmetric hoặc Depthwise tùy chọn)
        self.bottleneck = AsymmetricDepthwiseSeparableConv(self.enc_channels[3], self.enc_channels[3], stride=2) # Output: H/32, W/32, C=128

        # --- Bộ giải mã (Decoder) ---
        # Lưu trữ số kênh đầu ra của các tầng decoder
        self.dec_channels = [64, 32, 16, 16] # Số kênh sau mỗi khối decoder

        # Các khối giải mã với Skip Connections
        # Khối 1: Input từ bottleneck (128 kênh), skip từ enc_block3 (128 kênh)
        self.dec_block1 = DecoderBlockWithSkip(skip_channels=self.enc_channels[3], upsampled_channels=self.enc_channels[3], out_channels=self.dec_channels[0]) # Output: H/16, W/16, C=64
        # Khối 2: Input từ dec_block1 (64 kênh), skip từ enc_block2 (64 kênh)
        self.dec_block2 = DecoderBlockWithSkip(skip_channels=self.enc_channels[2], upsampled_channels=self.dec_channels[0], out_channels=self.dec_channels[1]) # Output: H/8, W/8, C=32
        # Khối 3: Input từ dec_block2 (32 kênh), skip từ enc_block1 (32 kênh)
        self.dec_block3 = DecoderBlockWithSkip(skip_channels=self.enc_channels[1], upsampled_channels=self.dec_channels[1], out_channels=self.dec_channels[2]) # Output: H/4, W/4, C=16
        # Khối 4: Input từ dec_block3 (16 kênh), skip từ entry_conv (16 kênh)
        self.dec_block4 = DecoderBlockWithSkip(skip_channels=self.enc_channels[0], upsampled_channels=self.dec_channels[2], out_channels=self.dec_channels[3]) # Output: H/2, W/2, C=16

        # --- Lớp đầu ra ---
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(self.dec_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Lưu lại các đầu ra của encoder để dùng làm skip connections
        skip_connections = []
        e0 = self.entry_conv(x)
        skip_connections.append(e0) # H/2, W/2, C=16
        # print("Sau entry_conv:", e0.shape)

        e1 = self.enc_block1(e0)
        skip_connections.append(e1) # H/4, W/4, C=32
        # print("Sau enc_block1:", e1.shape)

        e2 = self.enc_block2(e1)
        skip_connections.append(e2) # H/8, W/8, C=64
        # print("Sau enc_block2:", e2.shape)

        e3 = self.enc_block3(e2)
        skip_connections.append(e3) # H/16, W/16, C=128
        # print("Sau enc_block3:", e3.shape)

        # Bottleneck
        b = self.bottleneck(e3) # H/32, W/32, C=128
        # print("Sau bottleneck:", b.shape)

        # --- Decoder ---
        # Sử dụng các skip connections theo thứ tự ngược lại
        d1 = self.dec_block1(b, skip_connections.pop()) # Input: b (H/32, 128), Skip: e3 (H/16, 128) -> Output: H/16, 64
        # print("Sau dec_block1:", d1.shape)
        d2 = self.dec_block2(d1, skip_connections.pop()) # Input: d1 (H/16, 64), Skip: e2 (H/8, 64) -> Output: H/8, 32
        # print("Sau dec_block2:", d2.shape)
        d3 = self.dec_block3(d2, skip_connections.pop()) # Input: d2 (H/8, 32), Skip: e1 (H/4, 32) -> Output: H/4, 16
        # print("Sau dec_block3:", d3.shape)
        d4 = self.dec_block4(d3, skip_connections.pop()) # Input: d3 (H/4, 16), Skip: e0 (H/2, 16) -> Output: H/2, 16
        # print("Sau dec_block4:", d4.shape)

        # --- Output ---
        out = self.final_upsample(d4)
        # print("Sau final_upsample:", out.shape)
        out = self.final_conv(out)
        # print("Sau final_conv:", out.shape)

        return out

# --- Hàm kiểm tra số lượng tham số ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Ví dụ sử dụng và kiểm tra ---
if __name__ == '__main__':
    num_classes = 21 # Ví dụ: số lớp cho tập dữ liệu PASCAL VOC
    input_height, input_width = 224, 224 # Kích thước ảnh đầu vào ví dụ

    # Khởi tạo mô hình
    model = TinySegNetWithSkip(num_classes=num_classes)

    # Tạo dữ liệu đầu vào giả
    dummy_input = torch.randn(1, 3, input_height, input_width) # Batch size = 1

    # Chạy thử mô hình
    try:
        output = model(dummy_input)
        print(f"Kích thước đầu vào: {dummy_input.shape}")
        print(f"Kích thước đầu ra: {output.shape}") # Phải là (Batch, num_classes, H, W)

        # Kiểm tra số lượng tham số
        total_params = count_parameters(model)
        print(f"Tổng số tham số của mô hình: {total_params:,}")

        if total_params < 300000:
            print("Số lượng tham số đạt yêu cầu (< 300,000).")
        else:
            print("CẢNH BÁO: Số lượng tham số VƯỢT QUÁ giới hạn 300,000.")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình chạy mô hình: {e}")
        # In ra cấu trúc để dễ debug nếu có lỗi kích thước
        print("\nCấu trúc mô hình:")
        print(model)
