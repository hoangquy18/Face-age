# Output tensor của backbone (chỉ phần body)

**Backbone** ở đây = mạng lõi (IR stack / `mobilenet_v2.features` / encoder ViT), **trước** các lớp MTLFace phía sau: `Conv2d 1×1` xuống 512 kênh, `AdaptiveAvgPool`, FSM, head embedding.

Tham chiếu shape: `conda activate moe`, input **`B×3×112×112`**.

---

## Bảng — output backbone thuần

| Backbone | Symbol trong code (gần đúng) | Shape output |
|----------|------------------------------|--------------|
| **IR** (`ir34` … `irse101`) | `x_5` sau `block4` | **`B×512×7×7`** |
| **MobileNetV2** | `body(x)` = `mobilenet_v2.features` | **`B×1280×4×4`** |
| **ViT-B/32** | Sau encoder, patch tokens reshape (trước `proj` 768→512) | **`B×768×7×7`** |

- Với input 112×112, IR có stride không gian tổng **16** → `112/16 = 7`.
- MobileNetV2 (torchvision) stride tổng **32** → `112/32 = 3.5` floor **4** → map **`4×4`**, **1280** kênh ở lớp cuối của `features`.
- ViT-B/32: ảnh được nội suy **224×224** rồi patch **32×32** → **`7×7`** patch, mỗi token **768** chiều (ViT-Base).

---

## Giải thích ngắn (MobileNetV2 & ViT)

**MobileNetV2** là CNN tối ưu cho thiết bị yếu: dùng **depthwise separable convolution** và **inverted residual** → ít tham số và FLOPs hơn conv 2D đầy đủ cùng quy mô.

**ViT-B/32** là **Vision Transformer**: cắt ảnh thành patch, embed rồi qua **các lớp self-attention + MLP** (không có khối conv residual kiểu ResNet). `/32` là kích thước patch 32 pixel.

---

## Sau backbone (chỉ để nối context repo)

MTLFace gom về tensor chung **`B×512×S×S`** với `S = input_size // 16` (112 → `S=7`): `proj`/`pool` (MobileNet, ViT), rồi FSM → embedding `B×512`. Phần đó **không** tính là output backbone.
