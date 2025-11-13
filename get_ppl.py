import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from typing import List, Tuple


def calculate_sequence_perplexity(
    model_name: str,
    context: str,
    target_text: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    文全体のperplexityを1回のフォワードで計算（teacher forcing）
    コンテキストは損失から除外し、ターゲットトークンのみで平均クロスエントロピーを計算
    """
    print(f"モデルをロード中: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    # 入力ID作成（コンテキスト + ターゲット）
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    input_ids = torch.tensor([context_ids + target_ids], device=device)

    # ラベルは入力と同じだが、コンテキスト部分は-100でマスク
    labels = input_ids.clone()
    labels[:, :len(context_ids)] = -100
    # PADがある場合は無視（Qwenは多くの場合不要だが安全のため）
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss  # ターゲット部分のみの平均CE

    ppl = math.exp(loss.item())
    return ppl


def main():
    # 使用例
    model_name = "Qwen/Qwen2-0.5B"  # 小さめのモデルから試す
    
    # 例1: 自然な続き（文全体PPL）
    print("=" * 60)
    print("例1: 自然な続き（文全体PPL）")
    print("=" * 60)
    context1 = "今日は天気が良いので、"
    target1 = "公園に散歩に行きました。"
    
    print(f"コンテキスト: '{context1}'")
    print(f"対象文: '{target1}'")
    print()
    
    ppl1 = calculate_sequence_perplexity(model_name, context1, target1)
    print(f"\nPerplexity: {ppl1:.2f}")
    
    # 例2: 不自然な続き（文全体PPL）
    print("\n" + "=" * 60)
    print("例2: 不自然な続き（文全体PPL）")
    print("=" * 60)
    context2 = "今日は天気が良いので、"
    target2 = "量子力学の方程式を解きました。"
    
    print(f"コンテキスト: '{context2}'")
    print(f"対象文: '{target2}'")
    print()
    
    ppl2 = calculate_sequence_perplexity(model_name, context2, target2)
    print(f"\nPerplexity: {ppl2:.2f}")
    
    print("\n" + "=" * 60)
    print("注: Perplexityが低いほど、モデルにとって自然な文章です。")
    print("=" * 60)


if __name__ == "__main__":
    main()

