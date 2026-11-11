import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from PIL import Image
from tqdm import tqdm

PAD_COLOR = (255, 255, 255)
def pad_square(image, color=PAD_COLOR):
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas



class EVACLIP:
    def __init__(self, device="cuda"):
        self.device = device
        # "facebook/dinov2-base" 
        model_name = ["facebook/dinov2-base", "BAAI/EVA-CLIP-8B"][1]
        model_dtype=[torch.float16, torch.float32][0]
        self.model_dtype = model_dtype
        if model_dtype == torch.float32:
            model_class = get_class_from_dynamic_module("modeling_evaclip.EvaCLIPModel", "BAAI/EVA-CLIP-8B")
            model_class._no_split_modules = ["EvaCLIPEncoderLayer"]
        self.model, infos = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            output_loading_info=True,
            device_map="auto" if model_dtype==torch.float32 else None,
        )
        print(infos)
        if hasattr(self.model, "text_projection"):
            del self.model.text_projection
        if hasattr(self.model, "text_model"):
            del self.model.text_model

        #HOTFIX - evaclip8b에 pe 오류있음
        '''
        for module in self.model.modules():
            if hasattr(module, "position_ids"):
                module.position_ids = torch.arange(module.position_ids.shape[-1]).unsqueeze(0)
        '''
        if model_dtype!=torch.float32:
            self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def transform(self, paths, batch_size=64):
        # paths: 이미지 경로 리스트. 반환: [len(paths), D] 정규화된 numpy 배열 (paths 순서 유지)
        outputs = []
        for start in tqdm(range(0, len(paths), batch_size)):
            images = [pad_square(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]]
            inputs = (
                self.processor(images=images, return_tensors="pt")
                .pixel_values.to(self.device)
                .to(self.model_dtype)
                #.half()
            )
            if hasattr(self.model, "encode_image"):
                scores = self.model.encode_image(inputs)
            else:
                scores = self.model(inputs).pooler_output
            scores = torch.nn.functional.normalize(scores.float(), dim=-1)
            outputs.append(scores.cpu().numpy())
        return np.concatenate(outputs, axis=0)


