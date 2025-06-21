import re

def extract_references(text):
    # 假设text为参考文献部分
    pattern = r'\[(\d+)\]\s+([^\n]+)'
    matches = re.findall(pattern, text)
    references = []
    for match in matches:
        ref_num, ref_text = match
        references.append({'ref_num': ref_num, 'ref_text': ref_text})
    return references

# 示例
reference_section = """
[1] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
[2] Brown, T., et al. "Language models are few-shot learners." NeurIPS 2020.
"""
refs = extract_references(reference_section)
for ref in refs:
    print(ref)