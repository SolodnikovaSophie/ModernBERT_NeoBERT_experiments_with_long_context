import logging
import sys
import json
import torch
import collections
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from config import parse_args
from memory import MemoryTracker
from metrics import get_best_score_for_question, bootstrap_ci
from data_utils import (iter_nq_examples, get_question_id, get_question_text,
                        get_doc_tokens, is_valid_span, extract_answer_text,
                        build_intra_doc_context)


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


@torch.no_grad()
def predict_answer_batched(model, tokenizer, questions, contexts, device, max_seq_length):
    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"


        encoded = tokenizer([q.split() for q in questions], contexts, is_split_into_words=True,
                            max_length=max_seq_length, truncation="only_second", padding=True, return_tensors="pt")
        
        # --- ДОБАВЛЕННЫЙ БЛОК ДЛЯ LONGFORMER ---
        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        for i in range(len(questions)):
            seq_ids = encoded.sequence_ids(i)
            for j, s_id in enumerate(seq_ids):
                # Даем глобальное внимание токенам вопроса (seq_id == 0)
                if s_id == 0:
                    global_attention_mask[i, j] = 1
            # Обязательно глобальное внимание на CLS токен (индекс 0)
            global_attention_mask[i, 0] = 1
            
        model_inputs = {k: v.to(device) for k, v in encoded.items()}
        # Передаем маску в модель, если это Longformer
        if "longformer" in model.config.model_type.lower():
            model_inputs["global_attention_mask"] = global_attention_mask.to(device)
        
        outputs = model(**model_inputs)

        start_logits, end_logits = outputs.start_logits.cpu(), outputs.end_logits.cpu()
        batch_predictions = []

        for i in range(len(questions)):
            seq_ids, word_ids = encoded.sequence_ids(i), encoded.word_ids(i)
            s_i, e_i = start_logits[i], end_logits[i]
            best_s, best_span = None, None

            s_idxs = torch.topk(s_i, k=min(20, len(s_i))).indices.tolist()
            e_idxs = torch.topk(e_i, k=min(20, len(e_i))).indices.tolist()

            for si in s_idxs:
                for ei in e_idxs:
                    if ei < si or seq_ids[si] != 1 or seq_ids[ei] != 1: continue
                    sw, ew = word_ids[si], word_ids[ei]
                    if sw is None or ew is None or ew < sw: continue
                    score = float(s_i[si] + e_i[ei])
                    if best_s is None or score > best_s:
                        best_s, best_span = score, (sw, ew)

            if best_span:
                batch_predictions.append(" ".join(contexts[i][best_span[0]:best_span[1] + 1]).strip())
            else:
                batch_predictions.append("")
        return batch_predictions
    except Exception as e:
        logging.error(f"Error during batch inference: {e}")
        return [""] * len(questions)


def main():
    config = parse_args()
    config.save()
    setup_logger(config.output_dir / "run_position_bias.log")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    except ValueError:
        from transformers import PreTrainedTokenizerFast
        logging.warning("Falling back to PreTrainedTokenizerFast")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(config.model_name) / "tokenizer.json"),
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
        )

    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    model.to(device).eval()

    # --- Настройки эксперимента ---
    TARGET_SAMPLES = 1500
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    final_results = []

    for window_size in config.window_sizes:
        for pos in positions:
            logging.info(f">>> Starting Experiment: Window={window_size}, Target Position={pos}")

            output_subdir = config.output_dir / str(window_size) / str(pos)
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Логика сбора примеров
            q_results, g_results, logs = collections.defaultdict(list), collections.defaultdict(list), []
            valid_count = 0

            # Прогресс-бар для текущей позиции
            pbar = tqdm(total=TARGET_SAMPLES, desc=f"Size {window_size} Pos {pos}")

            batch_q, batch_c, batch_meta = [], [], []

            # Итерируемся по документам, пока не наберем TARGET_SAMPLES
            for ex in iter_nq_examples(config.input_path):
                if valid_count >= TARGET_SAMPLES:
                    break

                try:
                    qid = get_question_id(ex)
                    q_txt = get_question_text(ex)
                    doc_toks = get_doc_tokens(ex)

                    ann = ex.get("annotations", [{}])[0]
                    if not ann.get("short_answers"): continue
                    sa = ann["short_answers"][0]

                    # Пытаемся вырезать окно (сдвиг)
                    ctx = build_intra_doc_context(tokenizer, q_txt, doc_toks,
                                                  sa["start_token"], sa["end_token"],
                                                  window_size, pos)

                    if ctx is None:
                        continue  # Документ не подошел по длине или ответ не влез в позицию

                    gold = extract_answer_text(doc_toks, sa["start_token"], sa["end_token"])

                    batch_q.append(q_txt)
                    batch_c.append(ctx)
                    batch_meta.append({"qid": qid, "gold": gold})

                    if len(batch_q) >= config.batch_size:
                        preds = predict_answer_batched(model, tokenizer, batch_q, batch_c, device, window_size)
                        for m, p in zip(batch_meta, preds):
                            q_results[m["qid"]].append(p)
                            g_results[m["qid"]].append(m["gold"])
                            logs.append({"qid": m["qid"], "gold": m["gold"], "pred": p, "pos": pos})
                            valid_count += 1
                            pbar.update(1)
                            if valid_count >= TARGET_SAMPLES: break
                        batch_q, batch_c, batch_meta = [], [], []

                except Exception as e:
                    logging.warning(f"Error processing example {get_question_id(ex)}: {e}")
                    continue

            pbar.close()

            # Обработка последнего неполного батча
            if batch_q and valid_count < TARGET_SAMPLES:
                preds = predict_answer_batched(model, tokenizer, batch_q, batch_c, device, window_size)
                for m, p in zip(batch_meta, preds):
                    q_results[m["qid"]].append(p)
                    g_results[m["qid"]].append(m["gold"])
                    logs.append({"qid": m["qid"], "gold": m["gold"], "pred": p, "pos": pos})
                    valid_count += 1
                    if valid_count >= TARGET_SAMPLES: break

            # Если мы прошли весь датасет, но не набрали 500 для больших окон (например, 8192)
            if valid_count < TARGET_SAMPLES:
                logging.warning(
                    f"Could only find {valid_count}/{TARGET_SAMPLES} valid examples for Size {window_size} Pos {pos}")

            # --- РАСЧЕТ МЕТРИК ---
            metrics_vals = {"em": [], "f1": [], "rec": [], "rouge": []}

            for qid in q_results:
                best = get_best_score_for_question(q_results[qid], list(set(g_results[qid])))
                metrics_vals["em"].append(best["exact_match"])
                metrics_vals["f1"].append(best["f1"])
                metrics_vals["rec"].append(best["token_recall"])
                metrics_vals["rouge"].append(best["rougeL"])

            def get_stats(vals):
                mean = sum(vals) / len(vals) * 100 if vals else 0
                ci = bootstrap_ci(vals, config.bootstrap_samples, config.confidence_level, config.seed)
                return mean, [ci[0] * 100, ci[1] * 100]

            em_m, em_ci = get_stats(metrics_vals["em"])
            f1_m, f1_ci = get_stats(metrics_vals["f1"])
            rec_m, rec_ci = get_stats(metrics_vals["rec"])
            rg_m, rg_ci = get_stats(metrics_vals["rouge"])

            summary = {
                "window_size": window_size, "position": pos, "n_samples": valid_count,
                "EM": em_m, "EM_CI": em_ci, "F1": f1_m, "F1_CI": f1_ci,
                "Recall": rec_m, "Recall_CI": rec_ci, "RougeL": rg_m, "RougeL_CI": rg_ci
            }
            final_results.append(summary)

            # Сохранение детальных предсказаний для текущего окна/позиции
            with open(output_subdir / "predictions_log.json", "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)

            logging.info(f"Done. Samples: {valid_count}, EM: {em_m:.2f}, F1: {f1_m:.2f}, RougeL: {rg_m:.2f}")

    # Сохранение итоговых метрик
    with open(config.output_dir / "position_bias_final.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    main()