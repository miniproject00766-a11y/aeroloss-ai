import os
import glob
import torch
import torchvision.models as models
from PIL import Image

weights = models.MobileNet_V3_Small_Weights.DEFAULT
domain_model = models.mobilenet_v3_small(weights=weights).eval()
categories = weights.meta['categories']
transform = weights.transforms()

TEXTILE_CATEGORIES = {
    'wool', 'dishrag', 'quilt', 'velvet', 'poncho', 'jersey', 'towel', 
    'doormat', 'bath towel', 'sweatshirt', 'sock', 'cardigan', 'cloak', 
    'shawl', 'pajama', 'linen', 'apron', 'rug', 'carpet', 'handkerchief', 
    'fur coat', 'stole', 'bonnet', 'feather boa', 'jean', 'suit', 
    'swimming trunks', 'diaper', 'trench coat', 'kimono', 'sleeping bag', 
    'miniskirt', 'sarong', 'brassiere', 'bib', 'pillow', 'curtain', 'mitten',
    'cellular telephone', 'coffee mug', 'cup', 'plate', 'dining table',
    'pizza', 'cheeseburger', 'hotdog', 'bagel', 'sandwich', 'ice cream'
}

def check_semantic_domain(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = domain_model(tensor)
        probs = torch.softmax(out, dim=1).squeeze()
        top5 = torch.topk(probs, 5).indices.tolist()
    
    top_matches = [(categories[i], probs[i].item()) for i in top5]
    textile_hits = [(cat, prob) for cat, prob in top_matches if cat.lower() in TEXTILE_CATEGORIES]
    textile_prob_sum = sum(p for _, p in textile_hits)
    
    top_cat, top_prob = top_matches[0]
    if top_cat.lower() in TEXTILE_CATEGORIES:
        return False, f'Semantic Guardrail Veto: Image matches non-blade category "{top_cat}" ({top_prob*100:.1f}%). Authentic turbine blades consist of industrial composite fiberglass.'
    if textile_prob_sum > 0.15:
        reasons = ', '.join([f'{c} ({p*100:.1f}%)' for c, p in textile_hits])
        return False, f'Semantic Guardrail Veto: High textile/fabric correlation detected [{reasons}].'
    
    return True, 'Passed semantic domain check.'

if __name__ == '__main__':
    print('Testing User Fabric:')
    print('  user_fabric_test.jpg ->', check_semantic_domain('test_uploads/user_fabric_test.jpg'))
    print('\nTesting Authentic Blade Patches:')
    for p in sorted(glob.glob('test_uploads/0*.jpg')):
        print(f'  {os.path.basename(p)} ->', check_semantic_domain(p))
