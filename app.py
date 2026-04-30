import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 图像预处理
def get_transform():
    return T.Compose([T.ToTensor()])

@st.cache_resource
def load_models(model_name):
    from torchvision.models.segmentation import FCN_ResNet50_Weights
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights

    # 使用 Weights 参数确保加载最新的预训练权重，避免 Hash 错误
    if model_name == "FCN (语义分割)":
        model = torchvision.models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    elif model_name == "Faster R-CNN (目标检测)":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif model_name == "Mask R-CNN (实例分割)":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    model.eval()
    return model

st.set_page_config(page_title="视觉感知综合实验室", layout="wide")
st.title("🔍 计算机视觉多任务演示系统")
st.markdown("通过 Vibe Coding 实现：FCN 语义分割、Faster R-CNN 目标检测与 Mask R-CNN 实例分割。")

# 侧边栏配置
st.sidebar.header("配置选项")
model_type = st.sidebar.selectbox("选择任务模型", 
                                ["Faster R-CNN (目标检测)", "FCN (语义分割)", "Mask R-CNN (实例分割)"])
confidence_threshold = st.sidebar.slider("置信度阈值 (检测/实例)", 0.1, 1.0, 0.5)
uploaded_file = st.sidebar.file_uploader("上传测试图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    input_tensor = get_transform()(img).unsqueeze(0) # 增加 Batch 维度
    
    col1, col2 = st.columns(2)
    col1.image(img, caption="原始图片", use_column_width=True)
    
    model = load_models(model_type)
    
    with st.spinner("模型推理中..."):
        with torch.no_grad():
            # 解决 KeyError: 0 的关键：区分字典输出与列表输出[cite: 1]
            raw_output = model(input_tensor)
            
            if model_type == "FCN (语义分割)":
                # FCN 返回的是 OrderedDict, 取 'out' 键后的第 0 张图[cite: 1]
                output = raw_output['out'][0]
            else:
                # 检测模型返回的是 List[Dict], 取第 0 个元素的字典[cite: 1]
                output = raw_output[0]
            
    # --- 4. 结果可视化逻辑 ---
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    if model_type == "Faster R-CNN (目标检测)":
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']
        for box, score in zip(boxes, scores):
            if score > confidence_threshold:
                rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                     fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
        col2.pyplot(fig)

    elif model_type == "FCN (语义分割)":
        # 将输出转换为类别索引[cite: 1]
        seg_map = output.argmax(0).cpu().numpy()
        # 使用颜色映射展示分割结果
        fig_seg, ax_seg = plt.subplots()
        ax_seg.imshow(seg_map, cmap='viridis')
        ax_seg.axis('off')
        col2.pyplot(fig_seg)

    elif model_type == "Mask R-CNN (实例分割)":
        masks = output['masks']
        scores = output['scores']
        # 合并所有高置信度的 Mask 
        combined_mask = np.zeros(img.size[::-1])
        for i, score in enumerate(scores):
            if score > confidence_threshold:
                mask = masks[i, 0].cpu().numpy()
                combined_mask = np.where(mask > 0.5, 1, combined_mask)
        
        fig_mask, ax_mask = plt.subplots()
        ax_mask.imshow(img)
        ax_mask.imshow(combined_mask, alpha=0.5, cmap='jet') # 叠加半透明蒙版
        ax_mask.axis('off')
        col2.pyplot(fig_mask)

    # --- 5. 性能对比展示 ---
    st.divider()
    st.subheader("📊 不同模型性能指标对比")
    perf_data = {
        "模型名称": ["FCN", "Faster R-CNN", "Mask R-CNN"],
        "核心任务": ["语义分割 (Semantic)", "目标检测 (Detection)", "实例分割 (Instance)"],
        "输出粒度": ["像素级分类 (不分个体)", "框级定位 (区分个体)", "像素级轮廓 (区分个体)"],
        "计算复杂度": ["中等", "较低 (仅回归框)", "最高 (回归框+掩码)"]
    }
    st.table(perf_data)