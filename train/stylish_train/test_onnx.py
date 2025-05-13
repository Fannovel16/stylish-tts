import onnxruntime as ort
import numpy as np
from text_utils import TextCleaner
from utils import length_to_mask
from config_loader import load_model_config_yaml
from sentence_transformers import SentenceTransformer
import torch

model_config = load_model_config_yaml("/content/stylish-tts/config/model.yml")
text_cleaner = TextCleaner(model_config.symbol)
sbert = SentenceTransformer(model_config.sbert.model).cpu()
texts = (
    torch.tensor(
        text_cleaner(
            'ciɲ↗ hit lɛzɤ↘,vɤj↗ năŋ xiəw↗ mi→ tʰwʷɤt↓ kuəʌ miɲ↘,da→ tɯ↓ tăj tʰem vaw↘ mot↓ cɯ→"van↓"ɤʌ jɯə→ hiɲ↘ ʈɔn↘,ʈɔŋm nen↘ dɔʌ,ʈen la↗ kɤ↘ kuəʌ daŋʌ.'
        )
    )
    .unsqueeze(0)
    .cuda()
)
text_lengths = torch.zeros([1], dtype=int).cuda()
text_lengths[0] = texts.shape[1].cuda()
text_mask = length_to_mask(text_lengths)
sentence_embedding = (
    torch.from_numpy(
        sbert.encode(
            [
                'chính hitler với năng khiếu mỹ thuật của mình đã tự tay thêm vào một chữ"vạn"ở giữa hình tròn trong nền đỏ trên lá cờ của đảng.'
            ],
            show_progress_bar=False,
        )
    )
    .float()
    .cuda()
)
# Load ONNX model
session = ort.InferenceSession(
    "stylish.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
outputs = session.run(
    None,
    {
        "texts": texts.cpu().numpy(),
        "text_lengths": text_lengths.cpu().numpy(),
        "text_mask": text_mask.cpu().numpy(),
        "sentence_embedding": sentence_embedding.cpu().numpy(),
    },
)
print("Model output:", outputs[0])
