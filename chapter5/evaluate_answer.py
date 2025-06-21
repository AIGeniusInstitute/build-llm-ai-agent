from rouge import Rouge
import jieba  # 中文分词库


def evaluate_answer(pred, ref):
    # 对中文文本进行分词，并用空格连接
    pred_seg = ' '.join(jieba.cut(pred))
    ref_seg = ' '.join(jieba.cut(ref))

    rouge = Rouge()
    scores = rouge.get_scores(pred_seg, ref_seg)
    return scores


# 示例
pred = "Transformer模型引入了自注意力机制。"
ref = "自注意力机制是Transformer模型的核心创新。"

# 分词后的文本
pred_seg = ' '.join(jieba.cut(pred))
ref_seg = ' '.join(jieba.cut(ref))

print("分词后的预测文本:", pred_seg)
print("分词后的参考文本:", ref_seg)
print(evaluate_answer(pred, ref))