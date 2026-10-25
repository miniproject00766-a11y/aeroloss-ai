"""
Domain Guardrail Module for Wind Turbine Inspection Systems.
Implements a defensible, multi-signal evidence architecture:
1. Signal 1 (Image Integrity & Low-Level Quality):
   - Dimension check
   - Variance/Contrast check (rejects flat blank/black/white images)
   - Laplacian edge variance check (rejects untextured images and extreme white noise)
   - Color distribution / saturation check
2. Signal 2 (High-Level Semantic Out-of-Domain Filter):
   - Zero-shot ImageNet semantic rejection for non-industrial entities:
     * Biological organisms (animals, birds, insects, plants)
     * Food and produce
     * Personal consumer road vehicles
     * Apparel, textiles, garments
     * Synthetic documents, diagrams, charts, web pages, menus, receipts, paper goods
3. Signal 3 (Defect Classifier Signal Plausibility):
   - Entropy check on normalized prediction distribution
4. Decision Policy:
   - ACCEPT: High confidence in-domain turbine blade surface or inspection scene.
   - REJECT_OUT_OF_DOMAIN: Conclusive evidence that image is out-of-domain.
   - REVIEW_REQUIRED: Boundary condition, abnormal illumination, or uncertain semantic signals.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

class RobustDomainGuardrail:
    """
    Production-grade multi-signal Domain Guardrail with 3-way decision policy:
    ACCEPT, REJECT_OUT_OF_DOMAIN, REVIEW_REQUIRED.
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Semantic Foundation Filter (MobileNetV3 on ImageNet)
        try:
            self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=self.weights).to(self.device).eval()
            self.categories = self.weights.meta['categories']
            self.transform = self.weights.transforms()
        except Exception as e:
            print(f"RobustDomainGuardrail: Failed to load ImageNet backbone: {e}")
            self.model = None
            self.categories = []
            self.transform = None

        # Document, Diagram & UI Keywords (ImageNet classes that represent 2D documents/charts)
        self.DOCUMENT_DIAGRAM_KEYWORDS = {
            'web site', 'menu', 'envelope', 'packet', 'comic book', 'book jacket',
            'crossword puzzle', 'jigsaw puzzle', 'scoreboard', 'cash machine', 'vending machine',
            'screen, CRT screen', 'monitor', 'television, television receiver', 'oscilloscope', 'carton'
        }

        # Apparel & Textile Keywords
        self.TEXTILE_KEYWORDS = {
            'wool', 'dishrag', 'quilt', 'velvet', 'poncho', 'jersey', 'towel', 'rug',
            'carpet', 'cloth', 'fabric', 'suit', 'jacket', 'coat', 'blanket', 'linen',
            'curtain', 'bonnet', 'stole', 'shawl', 'lab coat', 'apron'
        }

        # Personal Electronics, Computing & Communication Devices
        self.ELECTRONICS_KEYWORDS = {
            'cellular telephone', 'cellular phone', 'cellphone', 'cell', 'mobile phone',
            'dial telephone', 'hand-held computer', 'laptop', 'notebook computer', 'ipod',
            'mouse, computer mouse', 'remote control', 'joystick', 'modem', 'keyboard',
            'tape player', 'cassette player', 'cd player', 'loudspeaker', 'microphone',
            'television', 'monitor', 'screen, crt screen'
        }

        # Musical Instruments (brass, woodwinds, strings, percussion, keyboards)
        self.MUSICAL_INSTRUMENT_KEYWORDS = {
            'trombone', 'trumpet', 'cornet', 'french horn', 'flute', 'oboe', 'sax', 'saxophone',
            'bassoon', 'violin', 'fiddle', 'cello', 'violoncello', 'guitar', 'acoustic guitar',
            'electric guitar', 'banjo', 'harp', 'piano', 'grand piano', 'upright, upright piano',
            'organ', 'drum', 'marimba', 'xylophone', 'chime', 'gong', 'harmonica', 'accordion'
        }

        # Consumer Accessories, Household Furnishings & Personal Items
        self.CONSUMER_ACCESSORY_KEYWORDS = {
            'sunglasses', 'sunglass', 'dark glasses', 'wallet', 'purse', 'backpack',
            'umbrella', 'shoe', 'sneaker', 'boot', 'sandal', 'sock', 'watch', 'analog clock',
            'digital clock', 'sofa', 'couch', 'bed', 'wardrobe', 'refrigerator', 'microwave',
            'toaster', 'blender', 'toilet', 'bathtub', 'pillow', 'plate', 'cup', 'mug'
        }

        # Natural Outdoor Backdrops Common Around Wind Farms (Alps, Cliffs, Sandbars)
        self.TURBINE_BACKDROPS = {970, 972, 975, 976, 977, 978, 979, 980}

    def evaluate(self, image_input, classifier_probs=None):
        """
        Evaluates input image against all domain safety criteria.
        Returns:
            status: 'ACCEPT' | 'REJECT_OUT_OF_DOMAIN' | 'REVIEW_REQUIRED'
            confidence: float (domain validity confidence 0.0 - 1.0)
            reason: str
            metrics: dict of low-level and high-level measurements
        """
        # Load PIL image
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, "Corrupt or unsupported image format.", {}

        w, h = pil_img.size
        if w < 32 or h < 32:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, f"Image dimensions too small ({w}x{h} < 32x32).", {}

        np_img = np.array(pil_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # -------------------------------------------------------------
        # SIGNAL 1: Low-Level Image Quality & Integrity Checks
        # -------------------------------------------------------------
        val_std = float(np.std(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        saturation = float(np.mean(np.max(np_img, axis=2) - np.min(np_img, axis=2)))

        metrics = {
            'image_size': [w, h],
            'intensity_std': round(val_std, 2),
            'laplacian_var': round(lap_var, 2),
            'saturation': round(saturation, 2)
        }

        # Check 1.1: Blank, Solid, or Low-Contrast Corrupt Files
        if val_std < 2.0:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, f"Blank or uniform image without surface texture (contrast std={val_std:.2f} < 2.0).", metrics

        # Check 1.2: Spatial Autocorrelation & Synthetic White/Gaussian Noise Check
        # Physical natural images (turbines, blades, skies) exhibit high spatial correlation (lag-1 r1 >= 0.70).
        # Uncorrelated random noise has lag-1 autocorrelation r1 near 0 (-0.05 to +0.10).
        gray_f = gray.astype(np.float32)
        gray_var = float(np.var(gray_f))
        if gray_var > 1e-3:
            mean_g = float(np.mean(gray_f))
            r1 = float(np.mean((gray_f[:, :-1] - mean_g) * (gray_f[:, 1:] - mean_g)) / gray_var)
        else:
            r1 = 0.0
        metrics['lag1_autocorrelation'] = round(r1, 4)

        if r1 < 0.30:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, f"Unnatural spatially uncorrelated pixel noise (lag-1 autocorrelation r1={r1:.3f} < 0.30).", metrics

        # Check 1.3: Synthetic Extreme High-Frequency Noise
        if lap_var > 20000.0:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, f"Unnatural extreme high-frequency noise (Laplacian var={lap_var:.1f} > 20,000).", metrics

        # Check 1.4: Flat untextured computer diagram or pure vector graphics
        if lap_var < 0.5 and val_std < 8.0:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, "Untextured solid color canvas.", metrics

        # Check 1.5: Synthetic neon color palette
        if saturation > 195.0:
            return 'REJECT_OUT_OF_DOMAIN', 0.0, f"Synthetic non-industrial color saturation ({saturation:.1f} > 195.0).", metrics

        # -------------------------------------------------------------
        # SIGNAL 2: Semantic Foundation ImageNet Filter
        # -------------------------------------------------------------
        top_name = "unknown"
        top_prob = 0.0
        if self.model is not None and self.transform is not None:
            try:
                t = self.transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(t)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    top5 = torch.topk(probs, 5)

                top_indices = top5.indices.tolist()
                top_probs = top5.values.tolist()
                top_idx, top_prob = top_indices[0], top_probs[0]
                top_name = self.categories[top_idx]
                metrics['semantic_top1'] = top_name
                metrics['semantic_top1_prob'] = round(float(top_prob), 4)

                # Check 2.1: Documents, Diagrams, Workflow Charts, Webpages, Infographics
                is_doc = any(k in top_name.lower() for k in self.DOCUMENT_DIAGRAM_KEYWORDS)
                doc_hits = [(self.categories[i].lower(), probs[i].item()) for i in top_indices if any(k in self.categories[i].lower() for k in self.DOCUMENT_DIAGRAM_KEYWORDS)]
                doc_sum = sum(p for _, p in doc_hits)
                if (is_doc and top_prob >= 0.20) or doc_sum >= 0.35:
                    return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Non-blade synthetic document or diagram detected ('{top_name}', {top_prob*100:.1f}%).", metrics

                # Check 2.2: Living Organisms (Birds, Animals, Insects, Plants: synsets 0..397)
                if top_idx not in self.TURBINE_BACKDROPS:
                    if 0 <= top_idx <= 397:
                        if top_prob >= 0.65:
                            return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Biological organism detected ('{top_name}', {top_prob*100:.1f}%). Authentic turbine blades are industrial composite structures.", metrics
                        elif top_prob >= 0.45:
                            return 'REVIEW_REQUIRED', 0.50, f"Uncertain visual signals resembling biological organism ('{top_name}', {top_prob*100:.1f}%). Manual inspection review required.", metrics

                # Check 2.3: Food, Produce & Beverages (synsets 924..969)
                if 924 <= top_idx <= 969:
                    PRODUCE_SHAPE_CONFUSION = {
                        'french loaf', 'burrito', 'cucumber', 'zucchini', 'spaghetti squash',
                        'acorn squash', 'butternut squash', 'banana', 'mashed potato', 'dough',
                        'bagel', 'pretzel', 'hay'
                    }
                    is_shape_confusion = any(k in top_name.lower() for k in PRODUCE_SHAPE_CONFUSION)
                    is_industrial_surface = (saturation <= 45.0 and r1 >= 0.70)

                    if not is_shape_confusion:
                        if top_prob >= 0.65:
                            return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Food or organic produce detected ('{top_name}', {top_prob*100:.1f}%).", metrics
                        elif top_prob >= 0.45:
                            return 'REVIEW_REQUIRED', 0.50, f"Uncertain visual signals resembling food/produce ('{top_name}', {top_prob*100:.1f}%). Manual inspection review required.", metrics
                    else:
                        if not is_industrial_surface and top_prob >= 0.75:
                            return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Food or organic produce detected ('{top_name}', {top_prob*100:.1f}%).", metrics
                        elif not is_industrial_surface and top_prob >= 0.45:
                            return 'REVIEW_REQUIRED', 0.50, f"Uncertain visual signals resembling produce ('{top_name}', {top_prob*100:.1f}%). Manual inspection review required.", metrics

                # Check 2.4: Consumer Road Transport / Vehicles
                if any(k in top_name.lower() for k in {'car', 'automobile', 'wagon', 'minivan', 'truck', 'bus', 'motorcycle', 'bicycle'}) and top_prob >= 0.35:
                    return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Consumer transport/vehicle detected ('{top_name}', {top_prob*100:.1f}%).", metrics

                # Check 2.5: Woven Apparel & Textiles
                textile_hits = [(self.categories[i].lower(), probs[i].item()) for i in top_indices if any(k in self.categories[i].lower() for k in self.TEXTILE_KEYWORDS)]
                if textile_hits:
                    hit_sum = sum(p for _, p in textile_hits)
                    top_is_textile = any(k in top_name.lower() for k in self.TEXTILE_KEYWORDS)
                    if (top_is_textile and top_prob >= 0.35) or hit_sum >= 0.40:
                        return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Woven fabric or apparel detected ('{top_name}', {top_prob*100:.1f}%). Industrial blades exhibit smooth aerodynamic gelcoat.", metrics
                    elif top_is_textile and top_prob >= 0.20:
                        return 'REVIEW_REQUIRED', 0.52, f"Surface texture resembles textile weave ('{top_name}', {top_prob*100:.1f}%). Manual verification recommended.", metrics

                # Check 2.6: Personal Consumer Electronics & Mobile Devices
                is_electronics = any(k in top_name.lower() for k in self.ELECTRONICS_KEYWORDS)
                elec_hits = [(self.categories[i].lower(), probs[i].item()) for i in top_indices if any(k in self.categories[i].lower() for k in self.ELECTRONICS_KEYWORDS)]
                if elec_hits:
                    elec_sum = sum(p for _, p in elec_hits)
                    if (is_electronics and top_prob >= 0.25) or elec_sum >= 0.35:
                        return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Personal electronic device or mobile phone detected ('{top_name}', {top_prob*100:.1f}%). Authentic turbine inspections capture industrial composite blade structures.", metrics
                    elif is_electronics and top_prob >= 0.15:
                        return 'REVIEW_REQUIRED', 0.50, f"Visual features resemble consumer electronics ('{top_name}', {top_prob*100:.1f}%). Manual inspection review required.", metrics

                # Check 2.7: Musical Instruments
                is_instrument = any(k in top_name.lower() for k in self.MUSICAL_INSTRUMENT_KEYWORDS)
                inst_hits = [(self.categories[i].lower(), probs[i].item()) for i in top_indices if any(k in self.categories[i].lower() for k in self.MUSICAL_INSTRUMENT_KEYWORDS)]
                if inst_hits:
                    inst_sum = sum(p for _, p in inst_hits)
                    if (is_instrument and top_prob >= 0.25) or inst_sum >= 0.35:
                        return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Musical instrument or non-industrial entity detected ('{top_name}', {top_prob*100:.1f}%). Authentic turbine inspections capture composite aerodynamic blade surfaces.", metrics
                    elif is_instrument and top_prob >= 0.15:
                        return 'REVIEW_REQUIRED', 0.50, f"Visual signals resemble musical instrument ('{top_name}', {top_prob*100:.1f}%). Manual inspection review required.", metrics

                # Check 2.8: Consumer Accessories & Household Furnishings
                is_accessory = any(k in top_name.lower() for k in self.CONSUMER_ACCESSORY_KEYWORDS)
                acc_hits = [(self.categories[i].lower(), probs[i].item()) for i in top_indices if any(k in self.categories[i].lower() for k in self.CONSUMER_ACCESSORY_KEYWORDS)]
                if acc_hits:
                    acc_sum = sum(p for _, p in acc_hits)
                    if (is_accessory and top_prob >= 0.30) or acc_sum >= 0.40:
                        return 'REJECT_OUT_OF_DOMAIN', round(1.0 - top_prob, 2), f"Household or personal consumer item detected ('{top_name}', {top_prob*100:.1f}%).", metrics

            except Exception as e:
                metrics['semantic_error'] = str(e)

        # -------------------------------------------------------------
        # SIGNAL 3: Defect Classifier Prediction Plausibility (if passed)
        # -------------------------------------------------------------
        if classifier_probs is not None and len(classifier_probs) > 0:
            probs_arr = np.array(classifier_probs)
            probs_arr = np.clip(probs_arr, 1e-7, 1.0)
            entropy = -np.sum(probs_arr * np.log(probs_arr))
            max_ent = np.log(len(probs_arr))
            norm_entropy = float(entropy / max_ent) if max_ent > 0 else 0.0
            top1_c = float(np.max(probs_arr))
            metrics['norm_entropy'] = round(norm_entropy, 3)
            metrics['defect_top1'] = round(top1_c, 3)

            # Extremely high entropy across all 6 classes indicates pure random noise
            if norm_entropy > 0.96 and top1_c < 0.22:
                return 'REVIEW_REQUIRED', 0.45, f"High prediction uncertainty across defect classes (entropy={norm_entropy:.2f}). Surface features are ambiguous.", metrics

        # -------------------------------------------------------------
        # AMBIGUOUS BOUNDARY CHECK: REVIEW_REQUIRED vs ACCEPT
        # -------------------------------------------------------------
        # Check if the image shows borderline low texture / extreme fog / blur while not being rejected outright
        if lap_var < 0.60:
            return 'REVIEW_REQUIRED', 0.55, f"Low image edge definition or heavy blur (Laplacian var={lap_var:.2f} < 0.60). Manual inspection review advised.", metrics

        # Default high-confidence acceptance
        domain_conf = float(np.clip(0.85 + (val_std / 200.0) * 0.10, 0.75, 0.99))
        return 'ACCEPT', round(domain_conf, 2), "Valid wind turbine inspection surface.", metrics

if __name__ == '__main__':
    guard = RobustDomainGuardrail()
    print("RobustDomainGuardrail initialized.")
