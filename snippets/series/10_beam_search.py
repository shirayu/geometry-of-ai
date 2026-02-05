from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BeamHypothesis:
    """ビームサーチの仮説（候補系列）"""

    tokens: list[int]  # トークン列
    score: float  # 対数確率の累積
    finished: bool = False  # 終了フラグ


def beam_search(
    model,
    start_tokens: torch.Tensor,
    beam_width: int = 4,
    max_length: int = 50,
    eos_token_id: int = 2,
    length_penalty: float = 1.0,
):
    """ビームサーチによるデコーディング

    Args:
        model: 言語モデル（next token確率を返す）
        start_tokens: 開始トークン列 [seq_len]
        beam_width: ビーム幅
        max_length: 最大生成長
        eos_token_id: 終了トークンID
        length_penalty: 長さペナルティ（>1で長い系列を好む）

    Returns:
        best_hypothesis: 最良の仮説
        all_hypotheses: すべての完了した仮説
    """
    device = start_tokens.device

    # 初期仮説
    initial_hyp = BeamHypothesis(tokens=start_tokens.tolist(), score=0.0)
    active_hypotheses = [initial_hyp]
    finished_hypotheses = []

    for _step in range(max_length):
        if not active_hypotheses:
            break

        all_candidates = []

        for hyp in active_hypotheses:
            if hyp.finished:
                finished_hypotheses.append(hyp)
                continue

            # 現在の系列に対する次トークン確率を取得
            input_ids = torch.tensor([hyp.tokens], device=device)

            with torch.no_grad():
                logits = model(input_ids)  # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Top-k の候補を取得
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for log_prob, token_id in zip(topk_log_probs.tolist(), topk_indices.tolist(), strict=True):
                new_tokens = hyp.tokens + [token_id]
                new_score = hyp.score + log_prob
                is_finished = token_id == eos_token_id

                new_hyp = BeamHypothesis(tokens=new_tokens, score=new_score, finished=is_finished)
                all_candidates.append(new_hyp)

        # スコアでソートして上位を選択
        # 長さペナルティを適用
        def score_with_penalty(hyp):
            length = len(hyp.tokens)
            return hyp.score / (length**length_penalty)

        all_candidates.sort(key=score_with_penalty, reverse=True)
        active_hypotheses = all_candidates[:beam_width]

        # 完了した仮説を分離
        new_active = []
        for hyp in active_hypotheses:
            if hyp.finished:
                finished_hypotheses.append(hyp)
            else:
                new_active.append(hyp)
        active_hypotheses = new_active

    # 残りの仮説も完了扱いに
    finished_hypotheses.extend(active_hypotheses)

    # 最良の仮説を選択
    if finished_hypotheses:
        best = max(finished_hypotheses, key=lambda h: h.score / (len(h.tokens) ** length_penalty))
    else:
        best = BeamHypothesis(tokens=start_tokens.tolist(), score=float("-inf"))

    return best, finished_hypotheses


# ダミーモデル（デモ用）
class DummyLanguageModel:
    """デモ用のダミー言語モデル"""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        # ランダムなロジットを返す
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return logits


# 使用例
if __name__ == "__main__":
    model = DummyLanguageModel(vocab_size=100)
    start_tokens = torch.tensor([1])  # 開始トークン

    best, all_hyps = beam_search(model, start_tokens, beam_width=4, max_length=10, eos_token_id=2)

    print(f"Best hypothesis: {best.tokens}")
    print(f"Score: {best.score:.4f}")
    print(f"Total hypotheses: {len(all_hyps)}")
