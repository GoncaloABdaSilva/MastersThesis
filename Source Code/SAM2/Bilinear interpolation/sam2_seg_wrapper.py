import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM2SegWrapper(nn.Module):
    def __init__(self, sam2_model, image_height=512, image_width=512):
        super().__init__()
        self.sam2_model = sam2_model
        self.image_height = image_height
        self.image_width = image_width
        self.mask_decoder = sam2_model.sam_mask_decoder
        self.image_encoder = sam2_model.image_encoder
        self.transformer_dim = self.mask_decoder.transformer_dim

    def forward(self, x):
        B = x.size(0)
        device = x.device

        # ---- Step 1: Encode image ----
        enc_out = self.image_encoder(x)  
        # enc_out['vision_features'] shape: [B, C, H_enc, W_enc]
        image_embeddings = enc_out['vision_features']
        image_pe = enc_out['vision_pos_enc']  # list of positional encodings

        C, H_enc, W_enc = image_embeddings.shape[1:]
        pos_enc = image_pe[-1]  # shape: [B, C, H_enc, W_enc]
        pos_enc = pos_enc[:1]  # [1, C, H_enc, W_enc]

        backbone_features = enc_out["backbone_fpn"]
        high_res_features = [backbone_features[-2], backbone_features[-1]]

        sparse_prompt_embeddings = torch.zeros((B, 0, self.transformer_dim), device=device)
        dense_prompt_embeddings = torch.zeros_like(image_embeddings)  # [B, C, H_enc, W_enc]

        img_emb = image_embeddings[:1]  # [1, C, H_enc, W_enc]

        self.mask_decoder.use_high_res_features = False

        # ---- Step 4: Forward through MaskDecoder ----
        mask_logits, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=pos_enc,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=True,
        )

        mask_logits = F.interpolate(mask_logits, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)

        return mask_logits
